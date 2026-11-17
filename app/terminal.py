"""Candle polling, ML qualification and opt-in automatic PAPER execution."""
from copy import deepcopy
from datetime import datetime
import json
import math
import threading
import time
import uuid

import joblib
from statsmodels.stats.multitest import multipletests

from app.market import PAIRS, IST, now, fresh, candle_frame
from ml.profit_model import train_profit, predict, features

DEFAULTS = dict(score=.65, allocation=100000., entry_z=2., stop_loss=.005,
                take_profit=.01, daily_loss=10000., max_minutes=25.)


class Terminal:
    def __init__(self, app):
        self.app=app
        self.models, self.candles, self.reports, self.signals = {}, {}, {}, {}
        self.screen=[]
        self.job={'running':False,'message':'Scan local, Upstox or Groww candles to train the candidate pairs.'}
        self.enabled=False
        self.config=deepcopy(DEFAULTS)
        self.last_candles=0.
        self.candle_time=None
        self.feed_error=None
        self.last_reasons={}
        self.generation=0
        with app.book.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS automation_events (id INTEGER PRIMARY KEY, account TEXT, time TEXT, kind TEXT, pair TEXT, message TEXT)')
            db.execute('CREATE TABLE IF NOT EXISTS automation_days (account TEXT, day TEXT, equity REAL, halted INTEGER DEFAULT 0, PRIMARY KEY(account,day))')
            db.execute('CREATE TABLE IF NOT EXISTS automation_settings (account TEXT PRIMARY KEY, body TEXT)')
        self.load_config()

    def load_config(self):
        with self.app.book.connect() as db:
            row=db.execute('SELECT body FROM automation_settings WHERE account=?',(self.app.account_key,)).fetchone()
        self.config={**DEFAULTS,**(json.loads(row[0]) if row else {})}

    def clock(self):
        return self.app.local.clock(self.app.account_key) if self.app.mode == 'demo' else now()

    def entry_decision(self, pair_id):
        decision = deepcopy(self.signals.get(pair_id) or {})
        if pair_id in self.candles and pair_id in self.reports:
            frame = candle_frame(self.candles[pair_id], completed_at=self.clock())
            row = features(frame, self.reports[pair_id]['baseline']).iloc[-1]
            if all(math.isfinite(float(v)) for v in row):
                decision['features'] = {k: float(v) for k, v in row.items()}
        decision['source'] = self.app.account_key
        return decision

    def event(self, kind, pair, message):
        with self.app.book.connect() as db:
            db.execute('INSERT INTO automation_events(account,time,kind,pair,message) VALUES(?,?,?,?,?)',
                       (self.app.account_key,self.clock().isoformat(),kind,pair,message))

    def reset(self):
        self.enabled=False
        self.app.local.running=False
        self.models.clear(); self.candles.clear(); self.reports.clear(); self.signals.clear()
        self.screen=[]; self.feed_error=None; self.last_candles=0.; self.candle_time=None
        self.job={'running':False,'message':'Scan this exchange to fit its models.'}
        self.generation+=1
        self.load_config()

    def can_switch(self, target=None):
        if hasattr(self.app,'demo_session') and self.app.demo_session.job['running']:
            raise ValueError('Stop the automated demo or wait for it to finish before switching accounts.')
        if self.app.local.batch_active:
            raise ValueError('Pause local advancement or wait for the batch before switching accounts.')
        if self.job['running']:
            raise ValueError('Wait for the current scan before switching data connections.')
        if target!=self.app.account_key and any(p.get('origin')=='auto' for p in self.app.book.read(self.app.account_key)['positions']):
            raise ValueError('Close automated positions before switching accounts; their exits need the current feed.')

    def scan(self):
        if self.app.local.batch_active:
            raise ValueError('Pause local advancement or wait for the batch before replacing models.')
        if self.job['running']:
            raise ValueError('A scan is already running.')
        if self.app.mode!='demo' and (not self.app.provider.token or getattr(self.app.provider, 'blocked', False)):
            raise ValueError('Connect your market-data provider before scanning real candles.')
        if any(p.get('origin')=='auto' for p in self.app.book.read(self.app.account_key)['positions']):
            raise ValueError('Close automated positions before replacing their models.')
        self.enabled=False
        self.models={}; self.candles={}; self.reports={}; self.signals={}; self.screen=[]
        self.app.local.running=False
        self.job={'running':True,'message':'Loading five-minute OHLCV and fitting models…'}
        generation=self.generation
        mode,exchange=self.app.mode,self.app.exchange
        training_clock=self.clock()
        local_bars={p['id']:self.app.local.candles(self.app.account_key,p['id']) for p in PAIRS} if mode=='demo' else {}
        def worker():
            models, candles, reports, rows={}, {}, {}, []
            try:
                for i,pair in enumerate(PAIRS):
                    with self.app.lock:
                        self.job['message']=f"{i+1}/{len(PAIRS)} · {pair['y']} / {pair['x']} · loading candles and training"
                    try:
                        bars=local_bars[pair['id']] if mode=='demo' else self.app.provider.candles(exchange,pair)
                        frame=candle_frame(bars, completed_at=training_clock)
                        model,report=train_profit(frame)
                        report.update(source=mode,exchange=exchange,trained_at=training_clock.isoformat())
                        models[pair['id']],candles[pair['id']],reports[pair['id']]=model,bars,report
                        rows.append(dict(id=pair['id'],name=f"{pair['y']} / {pair['x']}",p=report['baseline']['coint_p'],error=None))
                        stem=f"terminal-{mode}-{exchange}-{pair['id']}"
                        joblib.dump(model,self.app.directory/(stem+'.joblib'))
                        (self.app.directory/(stem+'.json')).write_text(json.dumps(report,allow_nan=False),encoding='utf-8')
                    except Exception as exc:
                        message=str(exc) if isinstance(exc,ValueError) else 'Data or model training failed. Retry the scan.'
                        rows.append(dict(id=pair['id'],name=f"{pair['y']} / {pair['x']}",p=1.,error=message))
                fdr=multipletests([r['p'] for r in rows],method='fdr_bh')[1]
                for row,q in zip(rows,fdr):
                    report=reports.get(row['id'])
                    reasons=[]
                    if row['error']: reasons.append(row['error'])
                    elif report:
                        if q>=.05: reasons.append('Cointegration FDR ≥ 0.05')
                        if report['baseline']['correlation']<.4: reasons.append('Return correlation < 0.40')
                        if not report['validation_pass']: reasons.append('Insufficient profitable validation signals')
                        if (report['validation_auc'] or 0)<=.5: reasons.append('Validation AUC ≤ 0.50')
                    row.update(fdr=float(q),eligible=not reasons,reasons=reasons,report=report)
                with self.app.lock:
                    if generation!=self.generation: return
                    self.models,self.candles,self.reports,self.screen=models,candles,reports,rows
                    self.last_candles=time.monotonic(); self.candle_time=self.clock().isoformat(); self.feed_error=None
                    self.update_signals()
                    count=sum(r['eligible'] for r in rows)
                    self.job={'running':False,'message':f'{count} of {len(PAIRS)} candidate pairs qualified. Automation remains paused.'}
                    self.event('scan','',self.job['message'])
            finally:
                with self.app.lock:
                    if generation==self.generation and self.job['running']:
                        self.job={'running':False,'message':'Scan could not finish. Retry; automation is paused.'}
        threading.Thread(target=worker,daemon=True).start()

    def update_signals(self):
        for pair_id,bars in self.candles.items():
            if any(row['id'] == pair_id and not row['eligible'] for row in self.screen):
                self.signals.pop(pair_id, None)
                continue
            try:
                frame=candle_frame(bars, completed_at=self.clock())
                report=self.reports[pair_id]
                signal=predict(self.models[pair_id],frame,report['baseline'])
                signal['threshold']=max(self.config['score'],report['threshold'])
                signal['fresh']=0 <= self.clock().timestamp()-(signal['bar_time']+300) <= 90
                signal['trained_at']=report['trained_at']
                self.signals[pair_id]=signal
                if hasattr(self.app, 'paper_learner') and self.app.account_key in self.app.paper_learner.models:
                    signal['paper_score']=self.app.paper_learner.score(self.entry_decision(pair_id).get('features', {}))
            except ValueError:
                self.signals.pop(pair_id,None)

    def tick(self):
        app=self.app
        with app.lock:
            generation=self.generation
            mode,exchange=app.mode,app.exchange
            ids=list(self.candles)
            provider=app.provider
            due=mode!='demo' and ids and not self.job['running'] and time.monotonic()-self.last_candles>=15
            if due: self.last_candles=time.monotonic()
        # Do not hold the application lock over network I/O: the terminal stays responsive.
        if due:
            try:
                updates={p['id']:provider.candles(exchange,p,intraday_only=True) for p in PAIRS if p['id'] in ids}
                with app.lock:
                    if generation!=self.generation: return
                    for pair_id,bars in updates.items():
                        for leg in ('y','x'):
                            merged={r['time']:r for r in self.candles[pair_id][leg]}
                            merged.update({r['time']:r for r in bars[leg]})
                            self.candles[pair_id][leg]=[merged[t] for t in sorted(merged)][-3000:]
                    self.candle_time=now().isoformat(); self.feed_error=None
            except Exception as exc:
                with app.lock:
                    if generation==self.generation:
                        self.feed_error=str(exc) if isinstance(exc,ValueError) else 'Candle refresh failed. New entries paused.'
        with app.lock:
            if generation!=self.generation: return
            self.update_signals()

    def configure(self, data):
        enabled=data.get('enabled')
        if not isinstance(enabled,bool): raise ValueError('Specify whether automation is enabled.')
        config=deepcopy(self.config)
        limits={'score':(.5,.95),'allocation':(1000,500000),'entry_z':(1.5,3.5),
                'stop_loss':(.001,.05),'take_profit':(.001,.10),'daily_loss':(100,50000),'max_minutes':(5,25)}
        for name,(lo,hi) in limits.items():
            value=data.get(name,config[name])
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not lo<=value<=hi:
                raise ValueError(f'{name} must be between {lo} and {hi}.')
            config[name]=float(value)
        if enabled:
            if self.job['running'] or not any(r['eligible'] for r in self.screen):
                raise ValueError('Scan first. At least one statistically and ML-qualified pair is required.')
            if self.app.error or self.feed_error: raise ValueError('Resolve the feed error before enabling automation.')
            if not all(fresh(q, self.clock()) for q in self.app.quotes.values()) or not self.app.quotes:
                raise ValueError('Fresh exchange trades are required to enable automation.')
            if self.day_state()['halted']: raise ValueError('Daily loss limit was reached. Entries remain halted for this session.')
        self.config=config; self.enabled=enabled
        with self.app.book.connect() as db:
            db.execute('INSERT OR REPLACE INTO automation_settings VALUES(?,?)',(self.app.account_key,json.dumps(config)))
        self.event('control','', 'Automatic paper entries enabled.' if enabled else 'Entries paused; automatic exits remain active.')

    def day_state(self):
        day=self.clock().astimezone(IST).date().isoformat()
        equity=self.app.book.mark(self.app.book.read(self.app.account_key),self.app.quotes,self.clock())['equity']
        with self.app.book.connect() as db:
            db.execute('INSERT OR IGNORE INTO automation_days(account,day,equity,halted) VALUES(?,?,?,0)',(self.app.account_key,day,equity))
            anchor,halted=db.execute('SELECT equity,halted FROM automation_days WHERE account=? AND day=?',(self.app.account_key,day)).fetchone()
        return dict(day=day,pnl=equity-anchor,halted=bool(halted))

    def manage(self):
        app=self.app
        account=app.book.mark(app.book.read(app.account_key),app.quotes,self.clock())
        automated=[p for p in account['positions'] if p.get('origin')=='auto']
        if not self.enabled and not automated: return
        day=self.day_state()
        # A feed outage pauses fills, not risk management: exits retry when quotes recover.
        if app.error: return
        if day['pnl']<=-self.config['daily_loss']:
            with app.book.connect() as db:
                db.execute('UPDATE automation_days SET halted=1 WHERE account=? AND day=?',(app.account_key,day['day']))
            day['halted']=True; self.enabled=False
        clock=self.clock().astimezone(IST)
        clock_minutes=clock.hour*60+clock.minute
        for p in automated:
            if not p['fresh']: continue
            decision=p.get('decision') or {}
            age=(self.clock()-datetime.fromisoformat(p['opened'])).total_seconds()/60
            config=decision.get('risk',self.config)
            entry_gross=sum(abs(l['qty'])*l['fill'] for l in p['legs'])
            # Include estimated exit fee/slippage and already paid entry fees in exit triggers.
            net=p['unrealized']-p['entry_fee']-p['gross']*.0008
            ret=net/entry_gross
            reason=None
            if day['halted']: reason='Daily loss limit'
            elif clock.date()!=datetime.fromisoformat(p['opened']).astimezone(IST).date() or clock_minutes>=15*60+20: reason='Session square-off'
            elif ret<=-config['stop_loss']: reason='Stop loss'
            elif ret>=config['take_profit']: reason='Profit target'
            elif age>=config['max_minutes']: reason='Maximum holding time'
            else:
                frozen=decision.get('spread')
                if frozen:
                    y,x=p['legs'][0]['symbol'],p['legs'][1]['symbol']
                    z=(app.quotes[y]['price']-frozen['alpha']-frozen['beta']*app.quotes[x]['price']-frozen['mean'])/frozen['std']
                    if abs(z)>=4: reason='Spread stop'
                    elif (p['side']=='long' and z>=-.5) or (p['side']=='short' and z<=.5): reason='Spread reverted'
            if reason: self.close(p,reason)
        if not self.enabled or day['halted'] or self.job['running'] or self.feed_error: return
        if clock.weekday()>=5 or not 9*60+20<=clock_minutes<14*60+55: return
        for row in self.screen:
            if not row['eligible']: continue
            pair_id=row['id']; signal=self.signals.get(pair_id)
            if not signal or not signal['fresh'] or not 0<=self.clock().timestamp()-(signal['bar_time']+300)<=90: continue
            if app.mode!='demo' and (self.clock()-datetime.fromisoformat(signal['trained_at'])).total_seconds()>86400: continue
            pair=next(p for p in PAIRS if p['id']==pair_id)
            if any(p['pair']==pair_id for p in app.book.read(app.account_key)['positions']): continue
            if not self.config['entry_z']<=abs(signal['zscore'])<4 or signal['score']<signal['threshold']: continue
            if app.account_key in app.paper_learner.enabled and (signal.get('paper_score') is None or signal['paper_score']<.5): continue
            if not all(s in app.quotes and fresh(app.quotes[s], self.clock()) for s in (pair['y'],pair['x'])): continue
            info=self.reports[pair_id]['baseline']
            frame=candle_frame(self.candles[pair_id], completed_at=self.clock())
            spreads=frame.y-info['alpha']-info['beta']*frame.x
            spread=dict(alpha=info['alpha'],beta=info['beta'],mean=float(spreads.iloc[-60:].mean()),std=float(spreads.iloc[-60:].std()))
            if not math.isfinite(spread['std']) or spread['std']<=1e-9: continue
            z=(app.quotes[pair['y']]['price']-spread['alpha']-spread['beta']*app.quotes[pair['x']]['price']-spread['mean'])/spread['std']
            if not self.config['entry_z']<=abs(z)<4 or (z<0)!=(signal['side']=='long'): continue
            # Permanent per-candle idempotency prevents re-entry after exit or repeated polls.
            request=str(uuid.uuid5(uuid.NAMESPACE_URL,f"{app.account_key}/{pair_id}/{signal['bar_time']}"))
            try:
                before=len(app.book.read(app.account_key)['positions'])
                app.book.trade(app.account_key,app.quotes,request,pair=pair,side=signal['side'],budget=self.config['allocation'],beta=info['beta'],origin='auto',
                    decision={**self.entry_decision(pair_id),'risk':deepcopy(self.config),'spread':spread,'model_version':2}, as_of=self.clock())
                if len(app.book.read(app.account_key)['positions'])>before:
                    self.event('entry',pair_id,f"{signal['side']} spread · model score {signal['score']:.3f} · completed candle {signal['bar_time']}")
            except ValueError as exc:
                if self.last_reasons.get(pair_id)!=str(exc): self.event('blocked',pair_id,str(exc))
                self.last_reasons[pair_id]=str(exc)

    def close(self, position, reason):
        try:
            self.app.book.trade(self.app.account_key,self.app.quotes,str(uuid.uuid4()),close_id=position['id'],exit_reason=reason,as_of=self.clock())
            self.event('exit',position['pair'],reason)
        except ValueError as exc:
            if self.last_reasons.get(position['id'])!=str(exc): self.event('blocked',position['pair'],str(exc))
            self.last_reasons[position['id']]=str(exc)

    def view(self):
        with self.app.book.connect() as db:
            rows=db.execute('SELECT time,kind,pair,message FROM automation_events WHERE account=? ORDER BY id DESC LIMIT 100',(self.app.account_key,)).fetchall()
        qualified=[]
        for row in self.screen:
            if not row['eligible']: continue
            pair=next(p for p in PAIRS if p['id']==row['id'])
            qualified.append(dict(id=pair['id'],y=pair['y'],x=pair['x'],signal=self.signals.get(pair['id']),
                candles={leg:bars[-300:] for leg,bars in self.candles[pair['id']].items()},
                fdr=row['fdr'],report=self.reports[pair['id']]))
        return dict(qualified=qualified,screen=self.screen,job=self.job,enabled=self.enabled,config=self.config,
                    candle_time=self.candle_time,feed_error=self.feed_error,
                    events=[dict(zip(['time','kind','pair','message'],r)) for r in rows])
