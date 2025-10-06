# main.py
import os, time, math, threading, signal
from datetime import datetime, timezone
from collections import deque
from dataclasses import dataclass
from termcolor import colored
import pandas as pd
from flask import Flask, request, jsonify

# =============== ENV =================
SYMBOL              = os.getenv("SYMBOL","DOGE/USDT:USDT")
INTERVAL            = os.getenv("INTERVAL","15m")
LEVERAGE            = float(os.getenv("LEVERAGE","10"))
RISK_PCT            = float(os.getenv("RISK_PCT","60"))  # نسبة الرصيد المُخاطر به (تُقسم على 10x)
DECISION_EVERY_S    = int(os.getenv("DECISION_EVERY_S","30"))
KEEPALIVE_SECONDS   = int(os.getenv("KEEPALIVE_SECONDS","50"))
PORT                = int(os.getenv("PORT","5000"))
LIVE                = os.getenv("LIVE","false").lower()=="true"
FORCE_TV_ENTRIES    = os.getenv("FORCE_TV_ENTRIES","true").lower()=="true"
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL","").strip()

TP1_PCT             = float(os.getenv("TP1_PCT","0.40"))
TP1_CLOSE_FRAC      = float(os.getenv("TP1_CLOSE_FRAC","0.50"))
TRAIL_ACTIVATE_PCT  = float(os.getenv("TRAIL_ACTIVATE_PCT","0.60"))
ATR_MULT_TRAIL      = float(os.getenv("ATR_MULT_TRAIL","1.6"))
BREAKEVEN_AFTER_PCT = float(os.getenv("BREAKEVEN_AFTER_PCT","0.30"))
HOLD_TP_STRONG      = os.getenv("HOLD_TP_STRONG","true").lower()=="true"
HOLD_TP_ADX         = float(os.getenv("HOLD_TP_ADX","28"))
HOLD_TP_SLOPE       = float(os.getenv("HOLD_TP_SLOPE","0.50"))
SCALE_IN_ENABLED    = os.getenv("SCALE_IN_ENABLED","true").lower()=="true"
SCALE_IN_ADX_MIN    = float(os.getenv("SCALE_IN_ADX_MIN","22"))
SCALE_IN_SLOPE_MIN  = float(os.getenv("SCALE_IN_SLOPE_MIN","0.20"))
SCALE_IN_MAX_ADDS   = int(os.getenv("SCALE_IN_MAX_ADDS","3"))
MIN_TP_PERCENT      = float(os.getenv("MIN_TP_PERCENT","0.40"))
MOVE_3BARS_PCT      = float(os.getenv("MOVE_3BARS_PCT","0.8"))
SPIKE_ATR_MULT      = float(os.getenv("SPIKE_FILTER_ATR_MULTIPLIER","2.5"))

ADX_LEN             = int(os.getenv("ADX_LEN","14"))
ATR_LEN             = int(os.getenv("ATR_LEN","14"))

API_KEY             = os.getenv("BINGX_API_KEY","")
API_SECRET          = os.getenv("BINGX_API_SECRET","")

# =============== EXCHANGE (ccxt) ===============
import ccxt
exchange = ccxt.bingx({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'options': {'defaultType': 'swap'},
    'enableRateLimit': True
})

# محاولة ضبط الرافعة
def ensure_leverage():
    try:
        exchange.setLeverage(int(LEVERAGE), SYMBOL)
    except Exception:
        pass

# =============== Helpers ===============
def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def fmt(v, n=6):
    try:
        return f"{float(v):.{n}f}"
    except Exception:
        return str(v)

def bool_ico(b): return colored("●","green") if b else colored("●","red")

def to_bool(s):
    return str(s).lower() in ("1","true","yes","y","on")

# =============== Indicators (pure pandas) ===============
def rsi(series, length=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ema_up = up.ewm(alpha=1/length, adjust=False).mean()
    ema_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = ema_up / (ema_down + 1e-12)
    return 100 - (100 / (1 + rs))

def atr(df, length=14):
    h,l,c = df['high'], df['low'], df['close']
    prev_c = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

def adx_di(df, length=14):
    # صيغة كلاسيك DI+/DI- وADX
    high, low, close = df['high'], df['low'], df['close']
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm= ((down_move > up_move) & (down_move > 0)) * down_move
    tr = pd.concat([(high-low).abs(), (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1/length, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/length, adjust=False).mean() / (atr_ + 1e-12))
    minus_di= 100 * (minus_dm.ewm(alpha=1/length, adjust=False).mean() / (atr_ + 1e-12))
    dx = ( (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-12) ) * 100
    adx = dx.ewm(alpha=1/length, adjust=False).mean()
    return plus_di, minus_di, dx, adx, atr_

# =============== Smart Engine (كما اتفقنا) ===============
@dataclass
class MarketState:
    price: float
    rsi: float
    adx: float
    dx: float
    di_plus: float
    di_minus: float
    atr: float
    rf_bias: str
    bar: dict

@dataclass
class Position:
    side: str
    entry: float
    qty: float
    pnl: float
    bars_in_pos: int
    scaled_adds: int
    stop: float|None = None

@dataclass
class SmartCfg:
    tp1_pct: float = TP1_PCT
    tp1_close_frac: float = TP1_CLOSE_FRAC
    trail_activate_pct: float = TRAIL_ACTIVATE_PCT
    atr_mult_trail: float = ATR_MULT_TRAIL
    breakeven_after_pct: float = BREAKEVEN_AFTER_PCT
    hold_tp_strong: bool = HOLD_TP_STRONG
    hold_tp_adx: float = HOLD_TP_ADX
    hold_tp_slope: float = HOLD_TP_SLOPE
    scale_in_enabled: bool = SCALE_IN_ENABLED
    scale_in_adx_min: float = SCALE_IN_ADX_MIN
    scale_in_slope_min: float = SCALE_IN_SLOPE_MIN
    scale_in_max_adds: int = SCALE_IN_MAX_ADDS
    min_tp_percent: float = MIN_TP_PERCENT
    move_3bars_pct: float = MOVE_3BARS_PCT
    spike_filter_atr_multiplier: float = SPIKE_ATR_MULT

def is_bull_engulf(b1, b0):
    return (b1['c']<b1['o']) and (b0['c']>b0['o']) and (b0['c']>=b1['o']) and (b0['o']<=b1['c'])

def is_bear_engulf(b1, b0):
    return (b1['c']>b1['o']) and (b0['c']<b0['o']) and (b0['o']>=b1['c']) and (b0['c']<=b1['o'])

def is_hammer(b0):
    body=abs(b0['c']-b0['o']); rng=b0['h']-b0['l']+1e-12
    low_tail=(max(b0['o'],b0['c'])-b0['l']); high_tail=(b0['h']-max(b0['o'],b0['c']))
    return (low_tail>2*body) and (high_tail<0.35*rng)

def is_shooting_star(b0):
    body=abs(b0['c']-b0['o']); rng=b0['h']-b0['l']+1e-12
    high_tail=b0['h']-max(b0['o'],b0['c']); low_tail=min(b0['o'],b0['c'])-b0['l']
    return (high_tail>2*body) and (low_tail<0.35*rng)

def is_doji(b0):
    body=abs(b0['c']-b0['o']); rng=(b0['h']-b0['l'])+1e-12
    return body<=0.1*rng

class SmartEngine:
    def __init__(self, cfg: SmartCfg):
        self.cfg=cfg
        self.prev = {'dx':None,'adx':None,'+di':None,'-di':None}
        self.bars_against=0

    def slopes(self, st:MarketState):
        def s(k, val):
            p=self.prev[k]; self.prev[k]=val
            if p is None: return 0
            return 1 if val>p else (-1 if val<p else 0)
        return s('dx',st.dx), s('adx',st.adx), s('+di',st.di_plus), s('-di',st.di_minus)

    def spike(self, st:MarketState):
        rng = st.bar['h']-st.bar['l']
        return rng >= self.cfg.spike_filter_atr_multiplier * st.atr

    def trailing(self, st:MarketState, pos:Position):
        move = abs(st.price-pos.entry)/(pos.entry+1e-12)
        if move < self.cfg.trail_activate_pct: return None,"trail:inactive"
        off = self.cfg.atr_mult_trail*st.atr
        return (st.price-off if pos.side=='LONG' else st.price+off), f"trail@ATRx{self.cfg.atr_mult_trail}"

    def breakeven(self, st:MarketState, pos:Position):
        move = abs(st.price-pos.entry)/(pos.entry+1e-12)
        if move >= self.cfg.breakeven_after_pct:
            return pos.entry, "be:on"
        return None, "be:off"

    def tp1(self, st:MarketState, pos:Position):
        tgt = pos.entry*(1+self.cfg.tp1_pct) if pos.side=='LONG' else pos.entry*(1-self.cfg.tp1_pct)
        hit = st.price>=tgt if pos.side=='LONG' else st.price<=tgt
        if hit: return self.cfg.tp1_close_frac, tgt, "tp1:hit"
        return 0.0, tgt, "tp1:wait"

    def scale_in(self, st:MarketState, pos:Position, sdx, sadx, splus, sminus):
        if not self.cfg.scale_in_enabled or pos.scaled_adds>=self.cfg.scale_in_max_adds:
            return 0.0,"scale:off"
        trend_ok_long  = (st.di_plus>st.di_minus) and (splus>0) and (st.adx>=self.cfg.scale_in_adx_min)
        trend_ok_short = (st.di_minus>st.di_plus) and (sminus>0) and (st.adx>=self.cfg.scale_in_adx_min)
        slope_ok = (sdx>0 or sadx>0)
        if pos.side=='LONG' and trend_ok_long and slope_ok:  return 0.25,"scale+"
        if pos.side=='SHORT' and trend_ok_short and slope_ok:return 0.25,"scale+"
        return 0.0,"scale:no"

    def hold_or_bank(self, st:MarketState, prev_bar, pos:Position, sdx, sadx, splus, sminus):
        strong = st.adx>=self.cfg.hold_tp_adx
        bull_eng=is_bull_engulf(prev_bar,st.bar); bear_eng=is_bear_engulf(prev_bar,st.bar)
        hammer=is_hammer(st.bar); star=is_shooting_star(st.bar); doji=is_doji(st.bar)

        if self.cfg.hold_tp_strong and strong:
            if pos.side=='LONG' and (splus>0 or sdx>0): return "HOLD","hold:trendUp"
            if pos.side=='SHORT'and (sminus>0 or sdx>0): return "HOLD","hold:trendDown"

        if pos.side=='LONG' and (bear_eng or star):  return "BANK","bearish-candle"
        if pos.side=='SHORT'and (bull_eng or hammer):return "BANK","bullish-candle"
        if doji and sadx<0: return "BANK_PART","doji-weak"

        if self.bars_against>=2:
            self.bars_against=0
            return "BANK_PART","3bars-against"

        return "NEUTRAL","ok"

    def step(self, st:MarketState, prev_bar, pos:Position):
        notes=[]
        if self.spike(st):
            return 0.0, None, 0.0, "spike-filter"

        close_frac, tgt, n = self.tp1(st,pos); notes.append(n)
        be, n = self.breakeven(st,pos); notes.append(n)
        tr, n = self.trailing(st,pos);  notes.append(n)

        sdx,sadx,splus,sminus = self.slopes(st)

        # تتبع الشمعة ضد الاتجاه
        bull = st.bar['c']>st.bar['o']
        self.bars_against = (self.bars_against+1) if ((pos.side=='LONG' and not bull) or (pos.side=='SHORT' and bull)) else 0

        act, why = self.hold_or_bank(st,prev_bar,pos,sdx,sadx,splus,sminus); notes.append(why)
        if act=="BANK":       close_frac=max(close_frac,1.0)
        elif act=="BANK_PART":close_frac=max(close_frac,0.33)

        add_frac, n = self.scale_in(st,pos,sdx,sadx,splus,sminus); notes.append(n)

        # أقل ربح مقبول لو ضعف الترند فجأة
        min_gain = self.cfg.min_tp_percent
        min_long = pos.entry*(1+min_gain)
        min_short= pos.entry*(1-min_gain)

        new_stop=None
        if be is not None and tr is not None:
            new_stop = max(be,tr) if pos.side=='LONG' else min(be,tr)
        elif be is not None:  new_stop = be
        else:                 new_stop = tr

        if new_stop is not None:
            if pos.side=='LONG' and st.price<min_long: new_stop=max(new_stop,min_long)
            if pos.side=='SHORT'and st.price>min_short:new_stop=min(new_stop,min_short)

        return close_frac, new_stop, add_frac, ";".join(notes)

smart = SmartEngine(SmartCfg())

# =============== TV Webhook & Runtime State ===============
app = Flask(__name__)
_last_ping = time.time()
_tv_signal = {'action':None,'ts':0}   # {'action':'BUY'|'SELL', 'ts': unix}
lock = threading.Lock()

@app.route("/tv", methods=["POST"])
def tv():
    global _tv_signal
    action = (request.json or {}).get("action","").upper()
    if action not in ("BUY","SELL"):
        return jsonify({"ok":False,"msg":"action must be BUY or SELL"}), 400
    with lock:
        _tv_signal = {'action':action,'ts':time.time()}
    return jsonify({"ok":True,"received":action})

@app.route("/")
def root():
    return jsonify({"status":"ok","symbol":SYMBOL,"interval":INTERVAL,"live":LIVE,"time":now_utc()})

def keepalive_loop():
    global _last_ping
    while True:
        if time.time()-_last_ping > KEEPALIVE_SECONDS:
            print(colored("keepalive ok (200)", "cyan"))
            _last_ping = time.time()
        time.sleep(KEEPALIVE_SECONDS/2)

# =============== Trading Logic ===============
position: Position|None = None
balance_cache = 0.0
bars_cache = deque(maxlen=600)

def fetch_ohlcv():
    # نجيب 300 شمعة
    o = exchange.fetch_ohlcv(SYMBOL, timeframe=INTERVAL, limit=300)
    df = pd.DataFrame(o, columns=["ts","open","high","low","close","vol"])
    df["ts"]=pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

def compute_indicators(df:pd.DataFrame):
    r = rsi(df['close'],14)
    pdi, ndi, dx, adx, atr_ = adx_di(df, ADX_LEN)
    df["rsi"]=r; df["+di"]=pdi; df["-di"]=ndi; df["dx"]=dx; df["adx"]=adx; df["atr"]=atr_
    return df

def account_equity():
    global balance_cache
    try:
        bal = exchange.fetch_balance()
        usdt = bal.get("USDT",{}).get("total") or bal.get("USDT",{}).get("free") or 0
        balance_cache = float(usdt)
    except Exception:
        pass
    return balance_cache

def next_qty(price):
    eq = account_equity()
    # المخاطرة على 10x: الكمية الاسمية = (eq * RISK_PCT%) * LEVERAGE / price
    nominal = (eq * (RISK_PCT/100.0)) * LEVERAGE / max(price,1e-9)
    # تقريب بسيط
    return max(1.0, round(nominal, 0))

def place_order(side, qty, price=None, reduce_only=False):
    side_ccxt = "buy" if side=="LONG" else "sell"
    params={'reduceOnly': reduce_only}
    try:
        if LIVE:
            return exchange.create_order(SYMBOL, type="market", side=side_ccxt, amount=qty, params=params)
        else:
            print(colored(f"[SIM] place_order {side} qty={qty}", "yellow"))
    except Exception as e:
        print(colored(f"order error: {e}", "red"))

def close_fraction(frac):
    global position
    if not position: return
    qty = max(0.0, position.qty * frac)
    if qty<=0: return
    side = "SHORT" if position.side=="LONG" else "LONG"
    place_order(side, qty, reduce_only=True)
    position.qty -= qty
    if position.qty<=0.000001:
        position=None

def update_stop(new_stop):
    # في BingX عبر ccxt مش دايمًا فيه stop dynamic موحّد؛ بنستخدم منطق تقفيل يدوي عند الكسر.
    # هنا بنخزّنه محليًا ونتصرف لو اتكسر.
    global position
    if position:
        position.stop = float(new_stop)

def maybe_stop_hit(last_price):
    global position
    if not position or position.stop is None: return
    if (position.side=="LONG" and last_price <= position.stop) or (position.side=="SHORT" and last_price >= position.stop):
        print(colored(f"⚠️ STOP hit @ {fmt(last_price)} -> closing ALL", "red"))
        close_fraction(1.0)

def open_from_tv(last_price):
    global position, _tv_signal
    with lock:
        act = _tv_signal['action']; ts = _tv_signal['ts']
        # إشارة صالحة للشمعة الحالية فقط (دقيقة واحدة هامش)
        valid = (time.time()-ts) < 120
        _tv_signal = {'action':None,'ts':0}
    if not act or not valid:
        return
    side = "LONG" if act=="BUY" else "SHORT"
    qty = next_qty(last_price)
    print(colored(f"🟢 TV SIGNAL => {act} | qty@{LEVERAGE}x ≈ {qty} {SYMBOL.split('/')[0]}", "green" if act=="BUY" else "red"))
    place_order(side, qty)
    # إنشاء بوزيشن محلي
    entry=last_price
    global position
    position = Position(side=side, entry=entry, qty=qty, pnl=0.0, bars_in_pos=0, scaled_adds=0, stop=None)

def maybe_scale_in(frac, last_price):
    global position
    if not position or frac<=0: return
    add_qty = max(1.0, round(position.qty * frac, 0))
    place_order(position.side, add_qty)
    position.qty += add_qty
    position.scaled_adds += 1
    print(colored(f"➕ scale-in {frac*100:.0f}% -> +{add_qty}", "magenta"))

def log_indicators(row, spread_bps, mode="SMART"):
    print(colored(f"\n{SYMBOL} | {INTERVAL} | LIVE • {now_utc()}", "yellow"))
    print(colored("INDICATORS","cyan"))
    print(f"  {colored('Price','white')}: {fmt(row['close'])}   {colored('RF','white')} filt:—  hi={fmt(row['high'])}  lo={fmt(row['low'])}")
    print(f"  {colored('RSI(14)','white')}={fmt(row['rsi'],2)}    {colored('+DI','white')}={fmt(row['+di'],2)}  {colored('-DI','white')}={fmt(row['-di'],2)}  "
          f"{colored('DX','white')}={fmt(row['dx'],2)}  {colored('ADX(14)','white')}={fmt(row['adx'],2)}  {colored('ATR','white')}={fmt(row['atr'],6)}")
    print(f"  spread_bps={fmt(spread_bps,2)}   Mode={mode}")

def log_position():
    if position:
        pnl = (position.qty*(last_close-position.entry)) if position.side=="LONG" else (position.qty*(position.entry-last_close))
        print(colored("POSITION","cyan"))
        print(f"  Balance {fmt(account_equity(),2)} USDT   Risk={int(RISK_PCT)}% x{int(LEVERAGE)}x   Cooldown=0")
        print(f"  {colored(position.side,'green' if position.side=='LONG' else 'red')}  Entry={fmt(position.entry)}  Qty={fmt(position.qty,0)}  Bars={position.bars_in_pos} "
              f" PnL={fmt(pnl,6)}  Stop={'-' if not position.stop else fmt(position.stop)}")
    else:
        print(colored("POSITION","cyan"))
        print("  FLAT")

def log_results(reason="no signal"):
    eq = account_equity()
    print(colored("RESULTS","cyan"))
    print(f"  CompoundPnL 0.000000   {colored('EffectiveEq','white')} {fmt(eq,2)} USDT")
    if position is None:
        print(f"  {colored('No trade','blue')} – reason: {reason} • 🕒")

# =============== Main Loop ===============
last_close = None

def run_loop():
    global position, last_close
    ensure_leverage()
    prev_bar = None

    while True:
        try:
            df = compute_indicators(fetch_ohlcv())
            row = df.iloc[-1]
            prev_row = df.iloc[-2]
            last_close = float(row["close"])

            # لوج المؤشرات
            log_indicators(row, spread_bps=1.2)

            # الدخول من TV فقط (لو مفعل)
            if position is None:
                if FORCE_TV_ENTRIES:
                    log_results("waiting tv")
                    open_from_tv(last_close)
                else:
                    # وضع يدوي/اختباري (اختياري)
                    log_results("manual/off")
            else:
                # إدارة الصفقة بالذكاء
                st = MarketState(
                    price=last_close,
                    rsi=float(row["rsi"]), adx=float(row["adx"]), dx=float(row["dx"]),
                    di_plus=float(row["+di"]), di_minus=float(row["-di"]),
                    atr=float(row["atr"]),
                    rf_bias="NEUTRAL",
                    bar={"o":float(row["open"]), "h":float(row["high"]), "l":float(row["low"]), "c":float(row["close"])}
                )
                prev_bar = {"o":float(prev_row["open"]), "h":float(prev_row["high"]), "l":float(prev_row["low"]), "c":float(prev_row["close"])}
                close_frac, new_stop, add_frac, note = smart.step(st, prev_bar, position)

                # تنفيذ القرارات
                if add_frac>0: maybe_scale_in(add_frac, last_close)
                if close_frac>=1.0: close_fraction(1.0)
                elif close_frac>0:  close_fraction(close_frac)
                if new_stop is not None: update_stop(new_stop)
                maybe_stop_hit(last_close)

                position.bars_in_pos += 1

                # لوج موجز للذكاء
                print(colored(f"🧠 SMART | {note} | stop={fmt(position.stop) if position and position.stop else '-'}", "white"))
                log_position()

        except Exception as e:
            print(colored(f"loop error: {e}", "red"))

        time.sleep(DECISION_EVERY_S)

# =============== Boot ===============
def handle_sigterm(signum, frame):
    print("Shutting down gracefully…")
    os._exit(0)

signal.signal(signal.SIGTERM, handle_sigterm)

if __name__ == "__main__":
    # خدمة ويب خفيفة + keepalive
    threading.Thread(target=keepalive_loop, daemon=True).start()
    threading.Thread(target=run_loop, daemon=True).start()

    # لو عندك Render External URL، هيبقى جاهز يستقبل /tv
    print(colored(f"Server up on :{PORT}  LIVE={LIVE}  TV_ONLY={FORCE_TV_ENTRIES}  URL={RENDER_EXTERNAL_URL or 'local'}","green"))
    app.run(host="0.0.0.0", port=PORT)
