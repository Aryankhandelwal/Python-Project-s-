# app.py
from flask import Flask, request, render_template_string
import yfinance as yf
import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)
portfolio = []  # simple in-memory portfolio

# Predefined stock list (you can add/remove tickers)
STOCK_LIST = ["RELIANCE.NS","TARIL.NS","TIL.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "ITC.NS", "RELIANCE.NS", "SBIN.NS", "BAJFINANCE.NS", "LT.NS"]

# ----------------- Helpers -----------------
def get_historical_prices(symbol, days=180):
    """Download daily history (reliable)"""
    try:
        df = yf.download(symbol, period=f"{days}d", interval="1d", progress=False)
        if df.empty:
            return None
        df = df[['Open','High','Low','Close','Volume']].rename(columns=str.lower)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return None

def get_latest_price(symbol):
    """Try to fetch the most recent market price robustly"""
    try:
        t = yf.Ticker(symbol)
        # fast_info is usually fastest
        if hasattr(t, "fast_info") and isinstance(t.fast_info, dict):
            last = t.fast_info.get("last_price") or t.fast_info.get("lastVolume")  # fallback
            if last:
                return float(last)
        # primary fallback: regularMarketPrice from info
        info = t.info if hasattr(t, "info") else {}
        rmp = info.get("regularMarketPrice") or info.get("previousClose")
        if rmp is not None:
            return float(rmp)
        # final fallback: recent close from history
        hist = t.history(period="5d")
        if not hist.empty:
            return float(hist['Close'].dropna().iloc[-1])
    except Exception:
        pass
    return None

def choose_benchmark(symbol):
    """Pick ^NSEI for Indian (.NS) tickers, otherwise ^GSPC (S&P500)"""
    s = symbol.upper()
    if s.endswith(".NS") or s.endswith(".BSE") or s.endswith(".BO") or s.endswith(".BE"):
        return "^NSEI"
    return "^GSPC"

def compute_beta(stock_df, market_df):
    try:
        df = pd.DataFrame({
            "stock": stock_df["close"].pct_change(),
            "market": market_df["close"].pct_change()
        }).dropna()
        if df.shape[0] < 10:
            return None
        cov = np.cov(df["stock"], df["market"])[0,1]
        var_market = np.var(df["market"])
        return float(cov / var_market) if var_market != 0 else None
    except Exception:
        return None

def format_large_number(value, currency_symbol=""):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    try:
        v = float(value)
    except Exception:
        return str(value)
    # nice units
    if abs(v) >= 1e12:
        return f"{currency_symbol}{v/1e12:,.2f} T"
    if abs(v) >= 1e9:
        return f"{currency_symbol}{v/1e9:,.2f} B"
    if abs(v) >= 1e6:
        return f"{currency_symbol}{v/1e6:,.2f} M"
    return f"{currency_symbol}{v:,.2f}"

def get_company_financials(symbol):
    """Return (latest_net_income, list_of_last4_quarters [(period_str, value)], market_cap, promoter_pct, currency)"""
    try:
        t = yf.Ticker(symbol)
        info = t.info if hasattr(t, "info") else {}
        currency = info.get("currency", "")
        currency_symbol = "₹" if currency == "INR" else ("$" if currency == "USD" else "")
        # annual net income (from financials)
        net_income = None
        try:
            fin = t.financials
            if not fin.empty and "Net Income" in fin.index:
                net_income = fin.loc["Net Income"].iloc[0]
        except Exception:
            net_income = None
        # quarterly: extract last 4 quarters net income
        quarterly_profits = []
        try:
            qfin = t.quarterly_financials
            if not qfin.empty and "Net Income" in qfin.index:
                s = qfin.loc["Net Income"]
                # take first 4 columns (yfinance usually orders most recent first)
                items = s.iloc[:4]
                for idx, val in items.items():
                    # idx may be Timestamp or str
                    try:
                        date_str = pd.to_datetime(idx).strftime("%b %Y")
                    except Exception:
                        date_str = str(idx)
                    quarterly_profits.append((date_str, val))
        except Exception:
            pass
        market_cap = info.get("marketCap")
        promoter_holding = info.get("heldPercentInsiders", None)
        return net_income, quarterly_profits, market_cap, promoter_holding, currency_symbol
    except Exception:
        return None, [], None, None, ""

# ---------------- HTML Template ----------------
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Portfolio & Stock Dashboard</title>
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <style>
        :root{--blue:#007bff;--dark:#222}
        body{font-family:Inter,Arial,Helvetica,sans-serif;background:#f5f7fb;margin:0;padding:20px;display:flex;flex-direction:column;align-items:center}
        .wrap{width:100%;max-width:1100px}
        header{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px}
        h1{margin:0;color:var(--dark)}
        form{display:flex;gap:8px;flex-wrap:wrap}
        select,input,button{padding:10px;border-radius:8px;border:1px solid #d6dbe8;font-size:14px}
        button{background:var(--blue);color:#fff;border:none;cursor:pointer}
        .card{background:#fff;border-radius:12px;padding:18px;box-shadow:0 6px 18px rgba(30,40,60,0.06);margin-bottom:16px}
        .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
        .metric{background:#fbfdff;padding:12px;border-radius:8px;border:1px solid #eef3fb}
        .metric strong{display:block;font-size:13px;color:#6b7280}
        .metric .val{font-size:18px;margin-top:6px;color:#111}
        table{width:100%;border-collapse:collapse;margin-top:12px}
        th,td{padding:10px;border-bottom:1px solid #eef1f6;text-align:center}
        th{background:linear-gradient(90deg,var(--blue),#0056b3);color:#fff}
        .green{color:green;font-weight:600}
        .red{color:red;font-weight:600}
        img{max-width:100%;border-radius:8px;margin-top:12px}
        @media (max-width:800px){
            .grid{grid-template-columns:repeat(1,1fr)}
        }
    </style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Portfolio & Stock Dashboard</h1>
  </header>

  <div class="card">
    <form method="POST" style="align-items:center;">
      <select name="symbol">
        {% for s in stock_list %}
          <option value="{{s}}" {% if result and result.symbol==s %}selected{% endif %}>{{s}}</option>
        {% endfor %}
      </select>
      <input type="text" name="custom" placeholder="Or type custom ticker (e.g. AAPL)" />
      <input type="number" name="quantity" placeholder="Qty (optional)" min="1" />
      <input type="number" step="0.01" name="price" placeholder="Buy Price (optional)" />
      <button type="submit">Analyze / Add</button>
    </form>
  </div>

  {% if result %}
  <div class="card">
    {% if result.error %}
      <div style="color:red;font-weight:600">{{ result.error }}</div>
    {% else %}
      <div style="display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap">
        <div>
          <h2 style="margin:0">{{ result.symbol }}</h2>
          <div style="margin-top:6px;color:#6b7280">{{ result.short_name }}</div>
        </div>
        <div style="text-align:right">
          <div style="font-size:24px;font-weight:700">{{ result.currency }}{{ result.latest_price }}</div>
          <div style="color:#6b7280;font-size:13px">Last updated live</div>
        </div>
      </div>

      <div class="grid" style="margin-top:14px">
        <div class="metric"><strong>Market Cap</strong><div class="val">{{ result.market_cap }}</div></div>
        <div class="metric"><strong>Beta (vs {{ result.benchmark }})</strong><div class="val">{{ result.beta_display }}</div></div>
        <div class="metric"><strong>Promoter / Insider Holding</strong><div class="val">{{ result.promoter }}</div></div>
      </div>

      {% if result.quarterly_profits %}
        <h3 style="margin-top:14px;margin-bottom:6px">Last 4 Quarters - Net Income</h3>
        <div class="grid" style="grid-template-columns:repeat(4,1fr)">
          {% for qlabel, qval in result.quarterly_profits %}
            <div class="metric"><strong>{{ qlabel }}</strong><div class="val">{{ qval }}</div></div>
          {% endfor %}
        </div>
      {% endif %}

      <img src="data:image/png;base64,{{ result.plot_url }}" alt="Price chart" />
    {% endif %}
  </div>
  {% endif %}

  <!-- Portfolio -->
  <div class="card">
    <h3 style="margin:0 0 10px 0">📈 Your Portfolio</h3>
    {% if portfolio %}
      <table>
        <tr><th>Stock</th><th>Qty</th><th>Buy Price</th><th>Current Price</th><th>Current Value</th><th>P/L</th></tr>
        {% for it in portfolio %}
          <tr>
            <td>{{ it.symbol }}</td>
            <td>{{ it.quantity }}</td>
            <td>{{ it.currency }}{{ it.buy_price }}</td>
            <td>{{ it.currency }}{{ it.current_price }}</td>
            <td>{{ it.currency }}{{ it.current_value }}</td>
            <td class="{{ 'green' if it.pnl>=0 else 'red' }}">{{ it.currency }}{{ it.pnl }}</td>
          </tr>
        {% endfor %}
      </table>
    {% else %}
      <div style="color:#6b7280">Your portfolio is empty. Add holdings using the form above.</div>
    {% endif %}
  </div>

</div>
</body>
</html>
"""

# ---------------- Routes ----------------
@app.route("/", methods=["GET","POST"])
def home():
    global portfolio
    result = None
    # POST: analyze / add
    if request.method == "POST":
        symbol = (request.form.get("custom") or request.form.get("symbol") or "").strip()
        symbol = symbol.upper()
        quantity = request.form.get("quantity")
        price = request.form.get("price")

        # validate
        if not symbol:
            result = {"error": "Please provide a ticker (select or type one)."}
        else:
            # get history and benchmark
            stock_hist = get_historical_prices(symbol, days=180)
            benchmark_symbol = choose_benchmark(symbol)
            market_hist = get_historical_prices(benchmark_symbol, days=180)

            if stock_hist is None:
                result = {"error": f"Could not fetch historical data for {symbol}. Check ticker."}
            else:
                # align date indexes for beta
                if market_hist is None:
                    beta = None
                else:
                    common = stock_hist.index.intersection(market_hist.index)
                    if len(common) < 30:
                        beta = None
                    else:
                        beta = compute_beta(stock_hist.loc[common], market_hist.loc[common])

                # generate chart (last 120 days)
                try:
                    buf = io.BytesIO()
                    plt.figure(figsize=(10,4))
                    plt.plot(stock_hist.index, stock_hist['close'], linewidth=2)
                    plt.title(f"{symbol} - Close Price")
                    plt.grid(alpha=0.25)
                    plt.tight_layout()
                    plt.savefig(buf, format="png", dpi=100)
                    plt.close()
                    buf.seek(0)
                    plot_url = base64.b64encode(buf.getvalue()).decode()
                except Exception:
                    plot_url = ""

                # get financials & last 4 quarters
                net_income, quarterly_profits_raw, market_cap_raw, promoter, currency_symbol = get_company_financials(symbol)
                # format quarterly profits as list of (label, formatted_value) for template
                quarterly_profits = []
                for label, val in (quarterly_profits_raw or []):
                    quarterly_profits.append((label, format_large_number(val, currency_symbol)))

                latest_price = get_latest_price(symbol)
                latest_price_disp = round(latest_price,2) if latest_price is not None else "N/A"

                # add to portfolio if qty & price provided
                if quantity and price:
                    try:
                        q = int(quantity)
                        p = float(price)
                        lp = latest_price if latest_price is not None else p
                        current_value = round(q * lp, 2)
                        pnl = round(current_value - (q * p), 2)
                        portfolio.append({
                            "symbol": symbol,
                            "quantity": q,
                            "buy_price": round(p,2),
                            "currency": currency_symbol,
                            # current values will be refreshed below before rendering
                        })
                    except Exception:
                        pass

                result = {
                    "symbol": symbol,
                    "short_name": symbol,
                    "latest_price": latest_price_disp,
                    "market_cap": format_large_number(market_cap_raw, currency_symbol),
                    "net_income": format_large_number(net_income, currency_symbol),
                    "quarterly_profits": quarterly_profits,
                    "promoter": f"{promoter*100:.2f}% " if promoter is not None else "N/A",
                    "beta": beta,
                    "beta_display": f"{beta:.2f}" if beta is not None else "N/A",
                    "plot_url": plot_url,
                    "benchmark": benchmark_symbol,
                    "currency": currency_symbol
                }

    # refresh portfolio current prices before rendering
    enriched_portfolio = []
    for item in portfolio:
        sym = item['symbol']
        q = item['quantity']
        buy = item['buy_price']
        cur_price = get_latest_price(sym)
        if cur_price is None:
            cur_price = 0.0
        cur_price = round(cur_price,2)
        cur_value = round(cur_price * q, 2)
        pnl = round(cur_value - (buy * q), 2)
        enriched_portfolio.append({
            "symbol": sym,
            "quantity": q,
            "buy_price": buy,
            "current_price": cur_price,
            "current_value": cur_value,
            "pnl": pnl,
            "currency": get_company_financials(sym)[4]  # currency symbol
        })

    return render_template_string(html_template, result=result, portfolio=enriched_portfolio, stock_list=STOCK_LIST)

if __name__ == "__main__":
    print("Server running on http://127.0.0.1:5000/")
    app.run(debug=True)
