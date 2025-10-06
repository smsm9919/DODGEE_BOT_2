# -*- coding: utf-8 -*-
"""
TradingView Smart Manager Pro — مشروع البوت
- Entry/Full Exit: EXACT TradingView Range Filter (BUY/SELL)
- In-Position Intelligence ONLY:
  * Classify trade: SCALP vs TREND (ATR%, ADX, DI, candles)
  * Smart TP1, Breakeven, ATR Trailing (ratchet), Give-Back
  * Candle AI: Doji/Pin/Engulfing + Explosive/False-Break handling
  * Partial-only exits; full close ONLY on opposite RF signal
- HUD: مؤشرات مُنظّمة + لمبة حالة 🟩 BUY / 🟥 SELL / 🟨 WAIT
- API: / (health), /metrics
"""

import os, time, math, threading, requests, traceback, random
import pandas as pd
import ccxt
from flask import Flask, jsonify
from datetime import datetime

# -------- console colors --------
try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t

# -------- ENV --------
def getenv(k, d=None, typ=str):
    v = os.getenv(k, d)
    if v is None: return d
    if typ is bool:  return str(v).lower() in ("1","true","yes","y","on")
    if typ is int:   return int(float(v))
    if typ is float: return float(v)
    return v

API_KEY    = getenv("BINGX_API_KEY", "")
API_SECRET = getenv("BINGX_API_SECRET", "")
MODE_LIVE  = bool(API_KEY and API_SECRET)

SYMBOL     = getenv("SYMBOL", "DOGE/USDT:USDT")
INTERVAL   = getenv("INTERVAL", "15m")
LEVERAGE   = getenv("LEVERAGE", 10, int)
RISK_ALLOC = getenv("RISK_ALLOC", 0.60, float)

# Range Filter params
RF_SOURCE  = getenv("RF_SOURCE", "close").lower()
RF_PERIOD  = getenv("RF_PERIOD", 20, int)
RF_MULT    = getenv("RF_MULT", 3.5, float)
USE_TV_BAR = getenv("USE_TV_BAR", False, bool)  # False => last CLOSED bar

# Indicators
RSI_LEN = getenv("RSI_LEN", 14, int)
ADX_LEN = getenv("ADX_LEN", 14, int)
ATR_LEN = getenv("ATR_LEN", 14, int)

# Execution guards
SPREAD_GUARD_BPS = getenv("SPREAD_GUARD_BPS", 6, int)
COOLDOWN_AFTER_CLOSE_BARS = getenv("COOLDOWN_AFTER_CLOSE_BARS", 0, int)

# Strategy switches (in-position only)
TP1_PCT        = getenv("TP1_PCT", 0.40, float)  # +0.40%: default TP1 if not TREND-HOLD
TP1_CLOSE_FRAC = getenv("TP1_CLOSE_FRAC", 0.50, float)
BREAKEVEN_AFTER= getenv("BREAKEVEN_AFTER_PCT", 0.30, float)
TRAIL_ACTIVATE = getenv("TRAIL_ACTIVATE_PCT", 0.60, float)
ATR_MULT_TRAIL = getenv("ATR_MULT_TRAIL", 1.6, float)

# Advanced in-position management (partials only; never full)
MIN_RUNNER_FRAC       = getenv("MIN_RUNNER_FRAC", 0.05, float)  # لا نغلق آخر 5% من الكمية
TRAIL_CLOSE_FRAC      = getenv("TRAIL_CLOSE_FRAC", 0.40, float) # نسبة الإغلاق عند ضرب الـ trail
BE_CLOSE_FRAC         = getenv("BE_CLOSE_FRAC", 0.50, float)    # نسبة الإغلاق عند لمس التعادل
GIVEBACK1_PEAK        = getenv("GIVEBACK1_PEAK", 1.20, float)   # % ربح أعلى لبدء حماية
GIVEBACK1_DROP        = getenv("GIVEBACK1_DROP", 0.60, float)   # % هبوط من القمة → إغلاق جزئي
GIVEBACK1_FRAC        = getenv("GIVEBACK1_FRAC", 0.25, float)
GIVEBACK2_PEAK        = getenv("GIVEBACK2_PEAK", 2.50, float)
GIVEBACK2_DROP        = getenv("GIVEBACK2_DROP", 1.00, float)
GIVEBACK2_FRAC        = getenv("GIVEBACK2_FRAC", 0.20, float)
SCALP_TP              = getenv("SCALP_TP", 0.30, float)         # % للصفقات الصغيرة
SCALP_DERISK_BARS     = getenv("SCALP_DERISK_BARS", 8, int)
SCALP_DERISK_THRESH   = getenv("SCALP_DERISK_THRESH", 0.20, float)
SCALP_DERISK_FRAC     = getenv("SCALP_DERISK_FRAC", 0.30, float)
EXPLOSIVE_ATR_MULT    = getenv("EXPLOSIVE_ATR_MULT", 1.8, float) # شمعة انفجارية
EXPLOSIVE_CLOSE_FRAC  = getenv("EXPLOSIVE_CLOSE_FRAC", 0.50, float)
REVERSAL_CLOSE_FRAC   = getenv("REVERSAL_CLOSE_FRAC", 0.35, float)

# pacing / keepalive
SLEEP_S  = getenv("DECISION_EVERY_S", 30, int)
SELF_URL = getenv("SELF_URL", "") or getenv("RENDER_EXTERNAL_URL","")
KEEPALIVE_SECONDS = getenv("KEEPALIVE_SECONDS", 50, int)
PORT     = getenv("PORT", 5000, int)

print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'} • SYMBOL={SYMBOL} • {INTERVAL}", "yellow"))

# -------- Exchange --------
def make_exchange():
    return ccxt.bingx({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType": "swap"}
    })

ex = make_exchange()
try:
    ex.load_markets()
except Exception as e:
    print(colored(f"⚠️ load_markets: {e}", "yellow"))

# -------- Helpers --------
def fmt(v, d=6, na="N/A"):
    try:
        if v is None or (isinstance(v,float) and (math.isnan(v) or math.isinf(v))): return na
        return f"{float(v):.{d}f}"
    except Exception: return na

def with_retry(fn, attempts=3, base_wait=0.4):
    for i in range(attempts):
        try: return fn()
        except Exception:
            if i == attempts-1: raise
            time.sleep(base_wait*(2**i) + random.random()*0.2)

def fetch_ohlcv(limit=600):
    rows = with_retry(lambda: ex.fetch_ohlcv(SYMBOL, timeframe=INTERVAL, limit=limit, params={"type":"swap"}))
    return pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])

def price_now():
    try:
        t = with_retry(lambda: ex.fetch_ticker(SYMBOL))
        return t.get("last") or t.get("close")
    except Exception as e:
        print(colored(f"❌ ticker: {e}", "red")); return None

def balance_usdt():
    if not MODE_LIVE: return 100.0
    try:
        b = with_retry(lambda: ex.fetch_balance(params={"type":"swap"}))
        return b.get("total",{}).get("USDT") or b.get("free",{}).get("USDT")
    except Exception as e:
        print(colored(f"❌ balance: {e}", "red")); return None

def orderbook_spread_bps():
    try:
        ob = with_retry(lambda: ex.fetch_order_book(SYMBOL, limit=5))
        bid = ob["bids"][0][0] if ob["bids"] else None
        ask = ob["asks"][0][0] if ob["asks"] else None
        if not (bid and ask): return None
        mid = (bid+ask)/2.0
        return ((ask-bid)/mid)*10000.0
    except Exception:
        return None

# -------- Indicators --------
def wilder_ema(s: pd.Series, n: int): return s.ewm(alpha=1/n, adjust=False).mean()

def compute_indicators(df: pd.DataFrame):
    c,h,l = df["close"].astype(float), df["high"].astype(float), df["low"].astype(float)
    tr = pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = wilder_ema(tr, ATR_LEN)

    delta = c.diff()
    up = delta.clip(lower=0.0); dn = (-delta).clip(lower=0.0)
    rs = wilder_ema(up, RSI_LEN) / wilder_ema(dn, RSI_LEN).replace(0,1e-12)
    rsi = 100 - (100/(1+rs))

    up_move = h.diff(); down_move = l.shift(1) - l
    plus_dm  = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    plus_di  = 100 * (wilder_ema(plus_dm, ADX_LEN) / atr.replace(0,1e-12))
    minus_di = 100 * (wilder_ema(minus_dm, ADX_LEN) / atr.replace(0,1e-12))
    dx = (100*(plus_di-minus_di).abs()/(plus_di+minus_di).replace(0,1e-12)).fillna(0.0)
    adx = wilder_ema(dx, ADX_LEN)

    i = len(df)-1 if USE_TV_BAR else len(df)-2
    return {"rsi": float(rsi.iloc[i]), "plus_di": float(plus_di.iloc[i]),
            "minus_di": float(minus_di.iloc[i]), "dx": float(dx.iloc[i]),
            "adx": float(adx.iloc[i]), "atr": float(atr.iloc[i])}

# -------- Range Filter (EXACT) --------
def _ema(s: pd.Series, n: int): return s.ewm(span=n, adjust=False).mean()
def _rng_size(src: pd.Series, qty: float, n: int) -> pd.Series:
    avrng = _ema((src - src.shift(1)).abs(), n); wper=(n*2)-1
    return _ema(avrng, wper)*qty
def _rng_filter(src: pd.Series, rsize: pd.Series):
    rf=[float(src.iloc[0])]
    for i in range(1,len(src)):
        prev=rf[-1]; x=float(src.iloc[i]); r=float(rsize.iloc[i]); cur=prev
        if x-r>prev: cur=x-r
        if x+r<prev: cur=x+r
        rf.append(cur)
    filt=pd.Series(rf,index=src.index,dtype="float64")
    return filt+rsize,filt-rsize,filt

def compute_tv_signals(df: pd.DataFrame):
    src=df[RF_SOURCE].astype(float)
    hi,lo,filt=_rng_filter(src,_rng_size(src,RF_MULT,RF_PERIOD))
    dfilt=filt-filt.shift(1)
    fdir=pd.Series(0.0,index=filt.index).mask(dfilt>0,1).mask(dfilt<0,-1).ffill().fillna(0.0)
    upward=(fdir==1).astype(int); downward=(fdir==-1).astype(int)
    src_gt_f=(src>filt); src_lt_f=(src<filt)
    src_gt_p=(src>src.shift(1)); src_lt_p=(src<src.shift(1))
    longCond=(src_gt_f & ((src_gt_p)|(src_lt_p)) & (upward>0))
    shortCond=(src_lt_f & ((src_lt_p)|(src_gt_p)) & (downward>0))
    CondIni=pd.Series(0,index=src.index)
    for i in range(1,len(src)):
        if bool(longCond.iloc[i]): CondIni.iloc[i]=1
        elif bool(shortCond.iloc[i]): CondIni.iloc[i]=-1
        else: CondIni.iloc[i]=CondIni.iloc[i-1]
    longSignal=longCond & (CondIni.shift(1)==-1)
    shortSignal=shortCond & (CondIni.shift(1)==1)
    i=len(df)-1 if USE_TV_BAR else len(df)-2
    return {"time": int(df["time"].iloc[i]), "price": float(df["close"].iloc[i]),
            "long": bool(longSignal.iloc[i]), "short": bool(shortSignal.iloc[i]),
            "filter": float(filt.iloc[i]), "hi": float(hi.iloc[i]), "lo": float(lo.iloc[i]),
            "fdir": float(fdir.iloc[i])}

# -------- Candles Intelligence --------
def _last(df, n=1): return df.iloc[-n]

def candle_stats(o,h,l,c):
    rng = max(h-l, 1e-12)
    body = abs(c-o)
    upper = h - max(c,o)
    lower = min(c,o) - l
    body_pct = (body/rng)*100.0
    upper_pct = (upper/rng)*100.0
    lower_pct = (lower/rng)*100.0
    dir_up = c >= o
    return {"range": rng, "body": body, "upper": upper, "lower": lower,
            "body_pct": body_pct, "upper_pct": upper_pct, "lower_pct": lower_pct,
            "up": dir_up}

def is_doji(s, max_body_pct=15): return s["body_pct"] <= max_body_pct
def is_pin_bull(s, min_lower_pct=55, max_body_pct=30): return s["lower_pct"]>=min_lower_pct and s["body_pct"]<=max_body_pct and s["up"]
def is_pin_bear(s, min_upper_pct=55, max_body_pct=30): return s["upper_pct"]>=min_upper_pct and s["body_pct"]<=max_body_pct and (not s["up"])
def engulfing_bull(o0,c0,o1,c1): return (c0<o0) and (c1>o1) and (o1<=c0) and (c1>=o0)
def engulfing_bear(o0,c0,o1,c1): return (c0>o0) and (c1<o1) and (o1>=c0) and (c1<=o0)

def analyze_candles(df, prev_high, prev_low):
    c1=_last(df,1); c0=_last(df,2)
    o1,h1,l1,cl1 = float(c1.open),float(c1.high),float(c1.low),float(c1.close)
    o0,h0,l0,cl0 = float(c0.open),float(c0.high),float(c0.low),float(c0.close)
    s1 = candle_stats(o1,h1,l1,cl1)
    patt=[]
    if is_doji(s1):     patt.append("DOJI")
    if is_pin_bull(s1): patt.append("PIN_BULL")
    if is_pin_bear(s1): patt.append("PIN_BEAR")
    if engulfing_bull(o0,cl0,o1,cl1): patt.append("ENGULF_BULL")
    if engulfing_bear(o0,cl0,o1,cl1): patt.append("ENGULF_BEAR")
    # انفجار: مدى الشمعة > n * ATR (يُحسب لاحقًا بتمرير ATR)
    return {"text":"|".join(patt) if patt else "NONE",
            "stats": s1,
            "last": {"o":o1,"h":h1,"l":l1,"c":cl1}}

# -------- BreakIntel + Regime --------
def compute_prev_levels(df, lookback=20):
    ph = float(df["high"].iloc[-(lookback+1):-1].max())
    pl = float(df["low"].iloc[-(lookback+1):-1].min())
    return ph, pl

def break_intel_signal(price, prev_high, prev_low, ema50, ema200, adx, pdi, mdi, atr):
    trend_up = ema50 > ema200
    trend_dn = ema50 < ema200
    break_res = price > prev_high + 0.2*atr
    break_sup = price < prev_low  - 0.2*atr
    if break_res and trend_up and adx>28 and pdi>mdi: return "WAIT_UP"
    if break_sup and trend_dn and adx>28 and mdi>pdi: return "WAIT_DOWN"
    return "NEUTRAL"

def detect_regime(ema50, ema200, adx, pdi, mdi, atr, price):
    atr_pct = (atr/max(price,1e-9))*100.0
    up  = (ema50>ema200) and (adx>=24) and (pdi>mdi)
    dn  = (ema50<ema200) and (adx>=24) and (mdi>pdi)
    micro = (adx<22) or (atr_pct<0.12)
    if up:   return "TREND_UP", micro
    if dn:   return "TREND_DOWN", micro
    return "RANGE", micro

def status_pill():
    if state["open"]:
        return colored("🟩 [BUY ACTIVE]", "green") if state["side"]=="long" else colored("🟥 [SELL ACTIVE]", "red")
    return colored("🟨 [WAIT]", "yellow")

# -------- State --------
state={
    "open":False,"side":None,"entry":None,"qty":0.0,"pnl":0.0,"bars":0,
    "trail":None,"tp1_done":False,"breakeven":None,
    "break_bias":"NEUTRAL","regime":"RANGE","micro":False,
    "peak_rr":0.0,"activated_trail":False,"trade_type":None,  # SCALP | TREND
    "init_qty":0.0,
    "candles":"NONE"
}
compound_pnl=0.0
post_close_cooldown=0

# -------- Orders --------
def compute_size(balance, price):
    capital = balance*RISK_ALLOC*LEVERAGE
    return max(0.0, capital/max(price,1e-9))

def partial_close_smart(frac, reason):
    """Partial close that NEVER fully closes the position (keep runner)."""
    global state, compound_pnl
    if not state["open"]: return
    min_runner = max(0.0, state["init_qty"]*MIN_RUNNER_FRAC)
    target_close = max(0.0, state["qty"]*min(max(frac,0.0),1.0))
    # لا نغلق آخر جزء
    max_allowed = max(0.0, state["qty"] - min_runner)
    qty_close = min(target_close, max_allowed)
    if qty_close <= 0: return
    px = price_now() or state["entry"]
    side = "sell" if state["side"]=="long" else "buy"
    if MODE_LIVE:
        try: ex.create_order(SYMBOL,"market",side,qty_close,params={"reduceOnly":True})
        except Exception as e: print(colored(f"❌ partial close: {e}","red"))
    pnl=(px-state["entry"])*qty_close*(1 if state["side"]=="long" else -1)
    compound_pnl+=pnl
    state["qty"]-=qty_close
    print(colored(f"🔻 PARTIAL {reason} closed={fmt(qty_close,4)} pnl={fmt(pnl)} rem_qty={fmt(state['qty'],4)}","magenta"))

def open_market(side, qty, price):
    global state
    if qty<=0:
        print(colored("❌ qty<=0 skip open","red")); return
    if MODE_LIVE:
        try: ex.set_leverage(LEVERAGE, SYMBOL, params={"side":"BOTH"})
        except Exception as e: print(colored(f"⚠️ set_leverage: {e}","yellow"))
        try: ex.create_order(SYMBOL,"market",side,qty,params={"reduceOnly":False})
        except Exception as e: print(colored(f"❌ open: {e}","red"))
    state={"open":True,"side":"long" if side=="buy" else "short","entry":price,"qty":qty,"pnl":0.0,"bars":0,
           "trail":None,"tp1_done":False,"breakeven":None,"break_bias":"NEUTRAL",
           "regime":"RANGE","micro":False,"peak_rr":0.0,"activated_trail":False,
           "trade_type":None,"init_qty":qty,"candles":"NONE"}
    print(colored(f"✅ OPEN {side.upper()} qty={fmt(qty,4)} @ {fmt(price)}","green" if side=="buy" else "red"))

def reset_after_full_close(reason):
    global state, post_close_cooldown
    print(colored(f"🔚 CLOSE reason={reason} totalCompounded now={fmt(compound_pnl)}","magenta"))
    state={"open":False,"side":None,"entry":None,"qty":0.0,"pnl":0.0,"bars":0,
           "trail":None,"tp1_done":False,"breakeven":None,"break_bias":"NEUTRAL",
           "regime":"RANGE","micro":False,"peak_rr":0.0,"activated_trail":False,
           "trade_type":None,"init_qty":0.0,"candles":"NONE"}
    post_close_cooldown = COOLDOWN_AFTER_CLOSE_BARS

def close_market(reason):
    """Full close — ONLY via opposite RF signal."""
    global state, compound_pnl
    if not state["open"]: return
    px=price_now() or state["entry"]; qty=state["qty"]
    side="sell" if state["side"]=="long" else "buy"
    if MODE_LIVE:
        try: ex.create_order(SYMBOL,"market",side,qty,params={"reduceOnly":True})
        except Exception as e: print(colored(f"❌ close: {e}","red"))
    pnl=(px-state["entry"])*qty*(1 if state["side"]=="long" else -1)
    compound_pnl+=pnl
    reset_after_full_close(reason)

# -------- Smart Exit (in-position only) --------
def smart_exit_check(info, ind, candle, prev_high, prev_low):
    if not state["open"]: return None
    px = info["price"]; e=state["entry"]; side=state["side"]
    rr = (px - e)/e * 100.0 * (1 if side=="long" else -1)
    atr = ind.get("atr") or 0.0
    adx = ind.get("adx") or 0.0

    # أول شمعتين: لا قرارات حادة
    if state["bars"] < 2:
        state["peak_rr"] = max(state["peak_rr"], rr)
        return None

    # تحديث peak
    state["peak_rr"] = max(state["peak_rr"], rr)

    # تصنيف الصفقة أول مرة: SCALP vs TREND
    if state["trade_type"] is None:
        atr_pct = (atr/max(px,1e-9))*100.0
        trendish = (adx>=24) and ((ind["plus_di"]>ind["minus_di"]) if side=="long" else (ind["minus_di"]>ind["plus_di"]))
        explosive = candle["stats"]["range"] >= EXPLOSIVE_ATR_MULT*atr
        state["trade_type"] = "TREND" if (trendish or explosive or atr_pct>=0.20) else "SCALP"
        print(colored(f"🧭 TRADE_TYPE={state['trade_type']} (ADX={fmt(adx,2)} ATR%={fmt(atr_pct,2)} Explosive={explosive})","cyan"))

    # ===== منطق SCALP =====
    if state["trade_type"]=="SCALP":
        if rr >= SCALP_TP and not state["tp1_done"]:
            partial_close_smart(0.60, f"SCALP_TP@{SCALP_TP:.2f}%")
            state["tp1_done"]=True
            if state["breakeven"] is None and rr >= BREAKEVEN_AFTER:
                state["breakeven"]=e
        if state["bars"] >= SCALP_DERISK_BARS and abs(rr) < SCALP_DERISK_THRESH:
            partial_close_smart(SCALP_DERISK_FRAC, "SCALP_TIME_DERISK")
            if state["breakeven"] is None: state["breakeven"]=e

    # ===== منطق TREND =====
    # BreakIntel: HOLD TP مبكر
    bias = state.get("break_bias","NEUTRAL")
    hold_bias = (side=="long" and bias=="WAIT_UP") or (side=="short" and bias=="WAIT_DOWN")
    if (not state["tp1_done"]) and rr >= TP1_PCT:
        if hold_bias:
            if state["breakeven"] is None and rr >= BREAKEVEN_AFTER: state["breakeven"]=e
            print(colored(f"🧠 BreakIntel HOLD_TP (bias={bias}, rr={fmt(rr,2)}%)","cyan"))
        else:
            partial_close_smart(TP1_CLOSE_FRAC, f"TP1@{TP1_PCT:.2f}%")
            state["tp1_done"]=True
            if state["breakeven"] is None and rr >= BREAKEVEN_AFTER: state["breakeven"]=e

    # شمعة انفجارية: اقفل جزء كبير فورًا
    rng = candle["stats"]["range"]
    if atr and rng >= EXPLOSIVE_ATR_MULT*atr:
        partial_close_smart(EXPLOSIVE_CLOSE_FRAC, f"EXPLOSIVE_RANGE>{EXPLOSIVE_ATR_MULT}xATR")

    # انعكاسات بالشموع
    note = state.get("candles","NONE")
    if side=="long":
        if ("ENGULF_BEAR" in note or "PIN_BEAR" in note) and rr>0.20:
            partial_close_smart(REVERSAL_CLOSE_FRAC, "CANDLE_REV_LONG")
            if state["breakeven"] is None: state["breakeven"]=e
        if "DOJI" in note and rr>0.30:
            partial_close_smart(0.20, "DOJI_LOCK")
    else:
        if ("ENGULF_BULL" in note or "PIN_BULL" in note) and rr>0.20:
            partial_close_smart(REVERSAL_CLOSE_FRAC, "CANDLE_REV_SHORT")
            if state["breakeven"] is None: state["breakeven"]=e
        if "DOJI" in note and rr>0.30:
            partial_close_smart(0.20, "DOJI_LOCK")

    # Trailing Ratchet (جزئي فقط)
    trail_on = (rr >= TRAIL_ACTIVATE) or state["activated_trail"]
    if trail_on and atr and ATR_MULT_TRAIL>0:
        state["activated_trail"] = True
        if rr >= 3.5:   mult = min(1.0, ATR_MULT_TRAIL)
        elif rr >= 2.0: mult = min(1.3, ATR_MULT_TRAIL)
        else:           mult = ATR_MULT_TRAIL
        gap = atr * mult
        if side=="long":
            new_trail = px - gap
            state["trail"] = max(state["trail"] or new_trail, new_trail)
            if state["breakeven"] is not None: state["trail"] = max(state["trail"], state["breakeven"])
            if px <= state["trail"]:
                partial_close_smart(TRAIL_CLOSE_FRAC, f"TRAIL_HIT({fmt(mult,2)}xATR)")
                # رفع trail بعد الإغلاق الجزئي
                state["trail"] = max(state["trail"], e)
        else:
            new_trail = px + gap
            state["trail"] = min(state["trail"] or new_trail, new_trail)
            if state["breakeven"] is not None: state["trail"] = min(state["trail"], state["breakeven"])
            if px >= state["trail"]:
                partial_close_smart(TRAIL_CLOSE_FRAC, f"TRAIL_HIT({fmt(mult,2)}xATR)")
                state["trail"] = min(state["trail"], e)

    # Give-Back protection (partials only)
    peak = state["peak_rr"]; loss_from_peak = peak - rr
    if peak >= GIVEBACK1_PEAK and loss_from_peak >= GIVEBACK1_DROP:
        partial_close_smart(GIVEBACK1_FRAC, "GIVEBACK_1")
        if state["breakeven"] is None: state["breakeven"]=e
    if peak >= GIVEBACK2_PEAK and loss_from_peak >= GIVEBACK2_DROP:
        partial_close_smart(GIVEBACK2_FRAC, "GIVEBACK_2")

    # Breakeven touch → partial only (لا غلق كامل)
    if state["breakeven"] is not None:
        if (side=="long" and px <= state["breakeven"]) or (side=="short" and px >= state["breakeven"]):
            partial_close_smart(BE_CLOSE_FRAC, "BREAKEVEN_TOUCH")

    return None

# -------- HUD --------
def snapshot(bal,info,ind,spread_bps,reason=None):
    print(colored("─"*100,"cyan"))
    print(colored(f"📊 {SYMBOL} {INTERVAL} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC","cyan"))
    print(colored("─"*100,"cyan"))
    print(colored("📈 INDICATORS","cyan"))
    print(colored(f"   💲 Price = {fmt(info.get('price'))}","white"))
    print(colored(f"   📊 RF    = filt {fmt(info.get('filter'))} | hi {fmt(info.get('hi'))} | lo {fmt(info.get('lo'))}","blue"))
    print(colored(f"   🌀 RSI({RSI_LEN}) = {fmt(ind['rsi'])}","cyan"))
    print(colored(f"   ➕ +DI      = {fmt(ind['plus_di'])}","green"))
    print(colored(f"   ➖ -DI      = {fmt(ind['minus_di'])}","red"))
    print(colored(f"   📉 DX       = {fmt(ind['dx'])}","magenta"))
    print(colored(f"   📡 ADX({ADX_LEN}) = {fmt(ind['adx'])}","yellow"))
    print(colored(f"   📏 ATR      = {fmt(ind['atr'])}","blue"))
    print(colored(f"   🧠 BreakIntel = {state.get('break_bias','NEUTRAL')}  |  📈 Regime = {state.get('regime','RANGE')} (micro={state.get('micro',False)})","cyan"))
    print(colored(f"   🕯️ Candles = {state.get('candles','NONE')}","white"))
    print(colored(f"   🧮 spread_bps = {fmt(spread_bps,2)}   BUY={info['long']}  SELL={info['short']}","white"))
    print()
    print("🧭 POSITION")
    print(f"   💰 Balance {fmt(bal,2)} USDT   Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x   Cooldown={post_close_cooldown}")
    if state["open"]:
        print(f"   📌 {'🟩 LONG' if state['side']=='long' else '🟥 SHORT'}  Entry={fmt(state['entry'])}  Qty={fmt(state['qty'],4)}  Bars={state['bars']}  PnL={fmt(state['pnl'])}")
        print(f"   🎯 Trail={fmt(state['trail'])}  TP1_done={state['tp1_done']}  BE={fmt(state['breakeven'])}  PeakRR={fmt(state['peak_rr'],2)}%  Type={state.get('trade_type')}")
    else:
        print("   ⚪ FLAT")
    print(f"   {status_pill()}")
    if reason:
        print(colored(f"   ℹ️ No trade — reason: {reason}","yellow"))
    print()
    print("📦 RESULTS")
    eff_eq = (bal or 0.0) + compound_pnl if MODE_LIVE else compound_pnl
    print(f"   🧮 CompoundPnL {fmt(compound_pnl)}   🚀 EffectiveEq {fmt(eff_eq)} USDT")
    print(colored("─"*100,"cyan"))

# -------- Decision Loop --------
def trade_loop():
    global state, post_close_cooldown, compound_pnl
    while True:
        try:
            bal=balance_usdt()
            px=price_now()
            df=fetch_ohlcv()
            info=compute_tv_signals(df)
            ind=compute_indicators(df)
            spread_bps = orderbook_spread_bps()

            if state["open"] and px:
                state["pnl"]=(px-state["entry"])*state["qty"] if state["side"]=="long" else (state["entry"]-px)*state["qty"]

            # BreakIntel + Regime
            prev_high, prev_low = compute_prev_levels(df, lookback=20)
            ema50 = df["close"].ewm(span=50, adjust=False).mean().iloc[-2]
            ema200= df["close"].ewm(span=200,adjust=False).mean().iloc[-2]
            bias = break_intel_signal(px or info["price"], prev_high, prev_low, ema50, ema200, ind["adx"], ind["plus_di"], ind["minus_di"], ind["atr"])
            state["break_bias"] = bias
            regime, micro = detect_regime(ema50, ema200, ind["adx"], ind["plus_di"], ind["minus_di"], ind["atr"], px or info["price"])
            state["regime"] = regime; state["micro"] = micro

            # Candles
            candle = analyze_candles(df, prev_high, prev_low)
            state["candles"] = candle["text"]

            # In-position smart management (partials only)
            smart_exit_check(info, ind, candle, prev_high, prev_low)

            # Decide by RF ONLY (زي TradingView)
            sig="buy" if info["long"] else ("sell" if info["short"] else None)
            reason=None
            if not sig:
                reason="no signal"
            elif spread_bps is not None and spread_bps>SPREAD_GUARD_BPS:
                reason=f"spread too high ({fmt(spread_bps,2)}bps > {SPREAD_GUARD_BPS})"
            elif post_close_cooldown>0:
                reason=f"cooldown {post_close_cooldown} bars"

            # Flip on opposite signal ALWAYS (full close then open)
            if state["open"] and sig:
                desired="long" if sig=="buy" else "short"
                if state["side"]!=desired and (reason is None):
                    close_market("opposite_signal")
                    # compute size and open new
                    qty=compute_size(bal, px or info["price"])
                    if qty>0:
                        open_market(sig, qty, px or info["price"])
                        snapshot(bal,info,ind,spread_bps,None)
                        time.sleep(SLEEP_S); continue

            # Open when flat
            if not state["open"] and (reason is None) and sig:
                qty=compute_size(bal, px or info["price"])
                if qty>0:
                    open_market(sig, qty, px or info["price"])
                else:
                    reason="qty<=0"

            snapshot(bal,info,ind,spread_bps,reason)

            if state["open"]:
                state["bars"] += 1
            if post_close_cooldown>0 and not state["open"]:
                post_close_cooldown -= 1

        except Exception as e:
            print(colored(f"❌ loop error: {e}\n{traceback.format_exc()}","red"))
        time.sleep(SLEEP_S)

# -------- Keepalive + API --------
def keepalive_loop():
    url = (SELF_URL or "").strip().rstrip("/")
    if not url:
        print(colored("⛔ keepalive: SELF_URL/RENDER_EXTERNAL_URL not set — skipping.", "yellow"))
        return
    sess = requests.Session()
    sess.headers.update({"User-Agent":"rf-pro-bot/keepalive"})
    print(colored(f"🛰️ keepalive: ping {url} every {KEEPALIVE_SECONDS}s","cyan"))
    while True:
        try:
            r = sess.get(url, timeout=8)
            if r.status_code==200: print(colored("🟢 keepalive ok (200)","green"))
            else: print(colored(f"🟠 keepalive status={r.status_code}","yellow"))
        except Exception as e:
            print(colored(f"🔴 keepalive error: {e}","red"))
        time.sleep(max(KEEPALIVE_SECONDS,15))

app = Flask(__name__)
@app.route("/")
def home():
    mode = 'LIVE' if MODE_LIVE else 'PAPER'
    return f"✅ مشروع البوت — {SYMBOL} {INTERVAL} — {mode} — TV Smart Manager Pro"

@app.route("/metrics")
def metrics():
    return jsonify({
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE,
        "risk_alloc": RISK_ALLOC,
        "price": price_now(),
        "position": state,
        "compound_pnl": compound_pnl,
        "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "strategy": "tv-smart-manager-pro"
    })

@app.route("/ping")
def ping(): return "pong", 200

# Boot
threading.Thread(target=trade_loop, daemon=True).start()
threading.Thread(target=keepalive_loop, daemon=True).start()
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
