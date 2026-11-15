"""Persistent, atomic two-leg simulator with separate demo/live-data accounts."""
from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime
import json
import math
import sqlite3
import uuid

from app.market import fresh, now

CAPITAL = 1_000_000.0
FEE = .0003
SLIPPAGE = .0005
BORROW = .10


class PaperBook:
    def __init__(self, path):
        self.path = str(path)
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS accounts (id TEXT PRIMARY KEY, body TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS requests (id TEXT PRIMARY KEY, account TEXT NOT NULL)")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=20)
        try:
            with db:
                yield db
        finally:
            db.close()

    def read(self, mode):
        with self.connect() as db:
            row = db.execute("SELECT body FROM accounts WHERE id=?", (mode,)).fetchone()
        return json.loads(row[0]) if row else {"capital": CAPITAL, "balance": CAPITAL, "positions": [], "trades": []}

    @staticmethod
    def mark(account, quotes, as_of=None):
        as_of = as_of or now()
        account = deepcopy(account)
        unrealized, gross = 0., 0.
        for p in account["positions"]:
            pnl, exposure = 0., 0.
            valid = True
            for leg in p["legs"]:
                q = quotes.get(leg["symbol"])
                price = q["price"] if q else leg["fill"]
                valid = valid and bool(q and fresh(q, as_of))
                pnl += leg["qty"] * (price - leg["fill"])
                exposure += abs(leg["qty"]) * price
            days = max(0, (as_of - datetime.fromisoformat(p["opened"])).total_seconds()/86400)
            borrow = sum(abs(l["qty"])*l["fill"] for l in p["legs"] if l["qty"] < 0)*BORROW*days/365
            p.update(unrealized=pnl-borrow, gross=exposure, borrow=borrow, fresh=valid)
            unrealized += pnl-borrow
            gross += exposure
        account.update(equity=account["balance"]+unrealized, unrealized=unrealized, gross=gross,
                       available=max(0, account["balance"]+unrealized-gross),
                       total_pnl=account["balance"]+unrealized-account["capital"])
        return account

    def trade(self, mode, quotes, request_id, pair=None, side=None, budget=None, beta=None, close_id=None,
              origin='manual', decision=None, exit_reason='Manual close', as_of=None):
        as_of = as_of or now()
        try:
            uuid.UUID(request_id)
        except (ValueError, TypeError, AttributeError):
            raise ValueError("A valid unique request ID is required.") from None
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            duplicate = db.execute("SELECT account FROM requests WHERE id=?", (request_id,)).fetchone()
            if duplicate:
                if duplicate[0] != mode:
                    raise ValueError("Request ID belongs to another account.")
                return
            row = db.execute("SELECT body FROM accounts WHERE id=?", (mode,)).fetchone()
            account = json.loads(row[0]) if row else self.read(mode)
            if close_id:
                p = next((p for p in account["positions"] if p["id"] == close_id), None)
                if p is None:
                    raise ValueError("Position is already closed or does not exist.")
                if as_of < datetime.fromisoformat(p['opened']):
                    raise ValueError('The simulation clock precedes this position. Advance local time to its entry date before closing.')
                symbols = [l["symbol"] for l in p["legs"]]
            else:
                if side not in ("long", "short") or not pair:
                    raise ValueError("Choose a valid pair and spread direction.")
                if not isinstance(budget, (int, float)) or not math.isfinite(budget) or not 1000 <= budget <= CAPITAL:
                    raise ValueError("Gross allocation must be between INR 1,000 and 10,00,000.")
                if not beta or not math.isfinite(beta) or beta <= 0:
                    raise ValueError("Load valid pair history before opening a position.")
                if any(p["pair"] == pair["id"] for p in account["positions"]):
                    raise ValueError("Close the existing position in this pair first.")
                symbols = [pair["y"], pair["x"]]
            if not all(s in quotes and fresh(quotes[s], as_of) for s in symbols):
                raise ValueError("Both legs require valid trades less than 60 seconds old. Fills paused.")
            times = [datetime.fromisoformat(quotes[s]["time"]) for s in symbols]
            if abs((times[0]-times[1]).total_seconds()) > 15:
                raise ValueError("Leg timestamps differ by more than 15 seconds. Fills paused.")
            if close_id:
                pnl, fee, fills = 0., 0., []
                marked = self.mark(account, quotes, as_of)
                borrow = next(x["borrow"] for x in marked["positions"] if x["id"] == close_id)
                for leg in p["legs"]:
                    fill = quotes[leg["symbol"]]["price"] * (1-SLIPPAGE if leg["qty"] > 0 else 1+SLIPPAGE)
                    pnl += leg["qty"] * (fill-leg["fill"])
                    fee += abs(leg["qty"])*fill*FEE
                    fills.append({"symbol": leg["symbol"], "qty": -leg["qty"], "fill": fill})
                account["balance"] += pnl-fee-borrow
                account["positions"].remove(p)
                account["trades"].append({"id": p["id"], "pair": p["pair"], "side": p["side"],
                    "origin": p.get('origin','manual'), "decision": p.get('decision'), "exit_reason": exit_reason,
                    "opened": p["opened"], "closed": as_of.isoformat(), "pnl": pnl-fee-borrow-p["entry_fee"],
                    "fees": fee+p["entry_fee"], "borrow": borrow, "entry_legs": p["legs"], "exit_legs": fills})
            else:
                marked = self.mark(account, quotes, as_of)
                if not all(p["fresh"] for p in marked["positions"]):
                    raise ValueError("Existing positions need fresh marks before increasing exposure.")
                if budget > marked["equity"]*.5 or budget*(1+FEE+2*SLIPPAGE) > marked["available"]:
                    raise ValueError("Allocation exceeds the 50% per-pair limit or available gross capacity.")
                py, px = [quotes[s]["price"] for s in symbols]
                units = budget / ((py+beta*px)*(1+SLIPPAGE))
                quantities = [math.floor(units), math.floor(units*beta)]
                if min(quantities) < 1:
                    raise ValueError("Allocation is too small for at least one share of each leg.")
                direction = 1 if side == "long" else -1
                legs, fee = [], 0.
                for symbol, qty in zip(symbols, [direction*quantities[0], -direction*quantities[1]]):
                    fill = quotes[symbol]["price"]*(1+SLIPPAGE if qty > 0 else 1-SLIPPAGE)
                    fee += abs(qty)*fill*FEE
                    legs.append({"symbol": symbol, "qty": qty, "fill": fill})
                account["balance"] -= fee
                account["positions"].append({"id": str(uuid.uuid4()), "pair": pair["id"], "side": side,
                    "origin": origin, "decision": decision,
                    "opened": as_of.isoformat(), "beta": beta, "entry_fee": fee, "legs": legs})
            db.execute("INSERT OR REPLACE INTO accounts VALUES (?,?)", (mode, json.dumps(account, allow_nan=False)))
            db.execute("INSERT INTO requests VALUES (?,?)", (request_id, mode))
