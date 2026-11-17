"""Daily report jobs and source-scoped artifacts, separate from paper execution."""
import json
from pathlib import Path
import threading
import uuid
import pandas as pd

from app.market import PAIRS, Upstox, demo_history, now
from app.groww import Groww


class ReportLab:
    def __init__(self,app):
        self.app=app
        self.directory=app.directory/'daily-report'
        self.directory.mkdir(exist_ok=True)
        self.results={}
        self.job=dict(running=False,message='Run the daily workflow from your project report.',account=None,error=None)
        for file in self.directory.glob('*/latest.json'):
            try:
                pointer=json.loads(file.read_text(encoding='utf-8'))
                run_id=pointer['run_id']
                uuid.UUID(run_id)
                result=json.loads((file.parent/run_id/'report.json').read_text(encoding='utf-8'))
                self.results[pointer['account']]=result
            except (OSError,ValueError,KeyError):continue

    def start(self):
        if self.job['running']:raise ValueError('A daily report run is already in progress.')
        app=self.app;account=app.account_key;mode,exchange=app.mode,app.exchange
        if mode!='demo' and (not app.provider.token or getattr(app.provider, 'blocked', False) or (mode=='live' and app.oauth_status()['expired'])):
            raise ValueError('Connect an active market-data session before loading daily exchange history.')
        provider=Upstox(app.provider.token) if mode=='live' else app.provider if mode=='groww' else None
        provider_name=app.provider_name
        self.job=dict(running=True,message='Loading daily prices for all 20 companies…',account=account,error=None)
        run_id=str(uuid.uuid4());directory=self.directory/account.replace(':','-')/run_id
        def progress(message):
            with app.lock:self.job['message']=message
        def worker():
            try:
                from ml.report.pipeline import run
                columns={}
                for i,pair in enumerate(PAIRS):
                    progress(f"Daily history {i+1}/{len(PAIRS)} · {pair['y']} / {pair['x']}")
                    frame=demo_history(i) if mode=='demo' else provider.history(exchange,pair)
                    for leg in ('y','x'):columns[pair[leg]]=frame[leg]
                source=f'{exchange} · synthetic generated daily prices' if mode=='demo' else f'{exchange} · {provider_name} daily historical closes (unadjusted)'
                result=run(pd.DataFrame(columns),directory,source,progress)
                result.update(run_id=run_id,account=account,created_at=now().isoformat())
                (directory/'report.json').write_text(json.dumps(result,allow_nan=False,indent=2),encoding='utf-8')
                pointer=directory.parent/'latest.json';temp=pointer.with_suffix('.tmp')
                temp.write_text(json.dumps(dict(run_id=run_id,account=account)),encoding='utf-8');temp.replace(pointer)
                with app.lock:
                    self.results[account]=result
                    self.job=dict(running=False,account=account,error=None,message=f"Completed: {result['verified']} verified pairs, {result['modeled']} fitted workflows, {len(result['backtest']['trades'])} holdout trades.")
            except Exception as exc:
                message=str(exc) if isinstance(exc,(ValueError,ImportError)) else 'Daily report failed. Check data/dependencies and retry; previous completed results were preserved.'
                with app.lock:self.job=dict(running=False,account=account,error=message,message=message)
        threading.Thread(target=worker,daemon=True).start()

    def view(self):
        return dict(job=self.job,result=self.results.get(self.app.account_key))

    def artifact(self,run_id,name):
        result=self.results.get(self.app.account_key)
        if not result or run_id!=result['run_id'] or name not in result['artifacts']+['report.json']:
            raise ValueError('Artifact not found for the active report account.')
        return self.directory/self.app.account_key.replace(':','-')/run_id/name
