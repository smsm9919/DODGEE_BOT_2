# مشروع البوت

بوت تداول DOGE/USDT (BingX Perp) بمنطق ذكاء متقدم:
- Range Filter (إشارات BUY/SELL).
- Smart Profit (TP1 جزئي + Breakeven + ATR Trailing).
- BreakIntel + Regime + Give-Back.
- Candles Intelligence (Doji/Pin/Engulfing + Fake Break).
- HUD ملون + /metrics + keepalive.

## التشغيل المحلي
```bash
pip install -r requirements.txt
export FLASK_ENV=production
python main.py
```

## متغيرات البيئة
انسخ `.env.example` إلى إعدادات Render (Environment) أو `.env` محليًا.

## Render
- اربط الريبو.
- استخدم `render.yaml`.
- ضع `SELF_URL` تلقائيًا من Render (Environment → RENDER_EXTERNAL_URL).
