"""FMP + yfinance wrappers. W0: byte-equivalent ports from seed v1."""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import requests

from src.config import FMP_BASE


def fetch_history(ticker, api_key, lookback_days):
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    url = f"{FMP_BASE}/historical-price-eod/full"
    params = {"symbol": ticker, "from": start, "to": end, "apikey": api_key}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"No FMP history for {ticker}")
    df = pd.DataFrame(data).rename(columns={"date": "Date", "close": "Close"})
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def _fmp_get(endpoint, api_key, params=None):
    p = {"apikey": api_key}
    if params:
        p.update(params)
    try:
        r = requests.get(f"{FMP_BASE}/{endpoint}", params=p, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"   WARNING: FMP {endpoint} failed: {e}")
        return None


def fetch_analyst_targets(ticker, api_key):
    """FMP price-target-consensus."""
    data = _fmp_get("price-target-consensus", api_key, {"symbol": ticker})
    if not data or not isinstance(data, list) or not data:
        return None
    d = data[0]
    return {
        "target_mean":   d.get("targetConsensus"),
        "target_median": d.get("targetMedian"),
        "target_high":   d.get("targetHigh"),
        "target_low":    d.get("targetLow"),
    }


def fetch_analyst_summary(ticker, api_key):
    """FMP price-target-summary — recent timeframe averages with analyst counts."""
    data = _fmp_get("price-target-summary", api_key, {"symbol": ticker})
    if not data or not isinstance(data, list) or not data:
        return None
    d = data[0]
    return {
        "last_month_count":   int(d.get("lastMonthCount", 0) or 0),
        "last_month_avg":     d.get("lastMonthAvgPriceTarget"),
        "last_quarter_count": int(d.get("lastQuarterCount", 0) or 0),
        "last_quarter_avg":   d.get("lastQuarterAvgPriceTarget"),
        "last_year_count":    int(d.get("lastYearCount", 0) or 0),
        "last_year_avg":      d.get("lastYearAvgPriceTarget"),
        "all_time_count":     int(d.get("allTimeCount", 0) or 0),
        "all_time_avg":       d.get("allTimeAvgPriceTarget"),
        "publishers":         d.get("publishers", ""),
    }


def fetch_next_earnings(ticker, api_key, lookahead_days=120):
    """FMP earnings-calendar — find next scheduled earnings event."""
    from_date = datetime.now().strftime("%Y-%m-%d")
    to_date = (datetime.now() + timedelta(days=lookahead_days)).strftime("%Y-%m-%d")
    data = _fmp_get("earnings-calendar", api_key,
                    {"from": from_date, "to": to_date})
    if not data or not isinstance(data, list):
        return None
    matches = [e for e in data if e.get("symbol") == ticker]
    if not matches:
        return None
    matches.sort(key=lambda x: x.get("date", "9999-99-99"))
    next_ev = matches[0]
    try:
        ev_date = datetime.strptime(next_ev["date"], "%Y-%m-%d")
        days_away = (ev_date.date() - datetime.now().date()).days
    except (ValueError, KeyError):
        return None
    return {
        "date": next_ev["date"],
        "days_away": days_away,
        "eps_est": next_ev.get("epsEstimated"),
        "rev_est": next_ev.get("revenueEstimated"),
        "in_horizon": False,
        "approaching": False,
    }


def fetch_company_profile(ticker, api_key):
    """FMP profile — sector, industry, market cap, etc."""
    data = _fmp_get("profile", api_key, {"symbol": ticker})
    if not data or not isinstance(data, list) or not data:
        return None
    return data[0]


def fetch_sector_perf(sector, api_key, days=30, exchange_filter="NASDAQ"):
    """FMP historical-sector-performance — filtered to one exchange, last-N unique dates."""
    if not sector:
        return None
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days + 14)).strftime("%Y-%m-%d")
    data = _fmp_get("historical-sector-performance", api_key,
                    {"sector": sector, "from": start, "to": end})
    if not data or not isinstance(data, list) or len(data) < 2:
        return None
    filtered = [r for r in data
                if not exchange_filter or r.get("exchange") == exchange_filter]
    if not filtered:
        filtered = data
    rows = sorted(filtered, key=lambda x: x.get("date", ""))
    rows = rows[-days:]
    if not rows:
        return None
    field_candidates = ["averageChange", "changesPercentage", "changePercent",
                        "change"]
    field = None
    for f in field_candidates:
        if f in rows[0]:
            field = f
            break
    if field is None:
        return None
    cum_return = 1.0
    for r in rows:
        try:
            val = float(r.get(field, 0))
            cum_return *= (1 + val / 100.0)
        except (ValueError, TypeError):
            continue
    cum_return -= 1.0
    return {
        "cum_return_pct": cum_return * 100,
        "n_days": len(rows),
        "sector": sector,
        "exchange": exchange_filter,
        "field_used": field,
    }


def fetch_insider_activity(ticker, api_key, days=90):
    """FMP insider-trading/search (canon endpoint)."""
    data = _fmp_get("insider-trading/search", api_key,
                    {"symbol": ticker, "limit": 100})
    if not data or not isinstance(data, list):
        return None
    cutoff = datetime.now() - timedelta(days=days)
    net_value = 0.0
    n_buys = 0
    n_sells = 0
    for tx in data:
        tx_type = (tx.get("transactionType") or "").upper()
        is_purchase = tx_type.startswith("P")
        is_sale = tx_type.startswith("S")
        if not (is_purchase or is_sale):
            continue
        try:
            tx_date = datetime.strptime(tx.get("transactionDate", "")[:10],
                                        "%Y-%m-%d")
            if tx_date < cutoff:
                continue
        except (ValueError, TypeError):
            continue
        try:
            shares = float(tx.get("securitiesTransacted", 0) or 0)
            price = float(tx.get("price", 0) or 0)
            value = shares * price
            if is_purchase:
                net_value += value
                n_buys += 1
            else:
                net_value -= value
                n_sells += 1
        except (ValueError, TypeError):
            continue
    return {
        "net_value_usd": net_value,
        "n_buys": n_buys,
        "n_sells": n_sells,
        "days": days,
    }


def fetch_recent_news(ticker, api_key, limit=20):
    """FMP news/stock — recent headlines."""
    data = _fmp_get("news/stock", api_key,
                    {"symbols": ticker, "limit": limit})
    if not data or not isinstance(data, list):
        return []
    out = []
    for item in data[:limit]:
        out.append({
            "date":      (item.get("publishedDate") or "")[:10],
            "publisher": item.get("publisher") or item.get("site") or "unknown",
            "title":     item.get("title") or "",
            "snippet":   (item.get("text") or "")[:200],
        })
    return out


def fetch_press_releases(ticker, api_key, limit=10):
    """FMP news/press-releases — official company releases."""
    data = _fmp_get("news/press-releases", api_key,
                    {"symbols": ticker, "limit": limit})
    if not data or not isinstance(data, list):
        return []
    out = []
    for item in data[:limit]:
        out.append({
            "date":    (item.get("publishedDate") or "")[:10],
            "title":   item.get("title") or "",
            "snippet": (item.get("text") or "")[:200],
        })
    return out


def fetch_macro_indicators(api_key):
    """FMP VIX + SPY for risk-on/risk-off."""
    vix_data = _fmp_get("quote", api_key, {"symbol": "^VIX"})
    spy_data = _fmp_get("quote", api_key, {"symbol": "SPY"})
    vix = 18.0
    spy_trend = 0.0
    if vix_data and isinstance(vix_data, list) and vix_data and vix_data[0].get("price"):
        vix = float(vix_data[0]["price"])
    if spy_data and isinstance(spy_data, list) and spy_data:
        d = spy_data[0]
        if d.get("price") and d.get("priceAvg50"):
            try:
                spy_trend = (float(d["price"]) - float(d["priceAvg50"])) / float(d["priceAvg50"])
            except (ValueError, TypeError, ZeroDivisionError):
                spy_trend = 0.0
    if vix > 25 or spy_trend < -0.03:
        regime = "risk_off"
    elif vix < 15 and spy_trend > 0.02:
        regime = "risk_on"
    else:
        regime = "neutral"
    return {"vix": vix, "spy_trend": spy_trend, "regime": regime}


def fetch_options_iv(ticker, target_dte_days=60):
    """yfinance options chain → ATM straddle IV at ~target_dte_days expiry.

    Liquidity-gated: only returns IV if option chain is liquid enough.
    """
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        tk = yf.Ticker(ticker)
        expiries = tk.options
        if not expiries:
            return None
        today = datetime.now().date()
        candidates = []
        for ex_str in expiries:
            try:
                ex_date = datetime.strptime(ex_str, "%Y-%m-%d").date()
                dte = (ex_date - today).days
                if 7 <= dte <= target_dte_days * 2:
                    candidates.append((abs(dte - target_dte_days), dte, ex_str))
            except ValueError:
                continue
        if not candidates:
            return None
        candidates.sort()
        _, dte, expiry = candidates[0]
        chain = tk.option_chain(expiry)
        spot = float(tk.fast_info.get("last_price", 0) or tk.history(period="1d")["Close"].iloc[-1])
        if spot <= 0:
            return None
        calls = chain.calls
        puts = chain.puts
        if calls.empty or puts.empty:
            return None
        atm_strike = float(calls.iloc[(calls["strike"] - spot).abs().argmin()]["strike"])
        atm_call = calls[calls["strike"] == atm_strike]
        atm_put = puts[puts["strike"] == atm_strike]
        if atm_call.empty or atm_put.empty:
            return None

        def spread_pct(row):
            bid = float(row["bid"])
            ask = float(row["ask"])
            mid = (bid + ask) / 2
            return abs(ask - bid) / mid if mid > 0 else 1.0

        call_spread = spread_pct(atm_call.iloc[0])
        put_spread = spread_pct(atm_put.iloc[0])
        avg_spread = (call_spread + put_spread) / 2
        is_liquid = avg_spread < 0.10
        call_iv = float(atm_call.iloc[0]["impliedVolatility"])
        put_iv = float(atm_put.iloc[0]["impliedVolatility"])
        avg_iv = (call_iv + put_iv) / 2
        return {
            "iv": avg_iv,
            "expiry": expiry,
            "dte": dte,
            "atm_strike": atm_strike,
            "bid_ask_pct_avg": avg_spread,
            "is_liquid": is_liquid,
            "call_iv": call_iv,
            "put_iv": put_iv,
        }
    except Exception as e:
        print(f"   WARNING: yfinance options IV fetch failed: {e}")
        return None


def fetch_short_interest(ticker, api_key):
    """yfinance short interest. FMP Starter lacks this endpoint."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        spf = info.get("shortPercentOfFloat")
        dtc = info.get("shortRatio")
        if spf is not None:
            return {
                "short_percent_of_float": float(spf),
                "days_to_cover": float(dtc) if dtc else None,
                "source": "yfinance",
            }
    except Exception as e:
        print(f"   WARNING: yfinance short-interest fetch failed: {e}")
    return None


def fetch_peer_history(peers, api_key, lookback_days=60):
    """Fetch closing-price history for peer tickers."""
    out = {}
    for p in peers:
        try:
            df = fetch_history(p, api_key, lookback_days=lookback_days)
            if df is not None and not df.empty:
                out[p] = df
        except Exception as e:
            print(f"   WARNING: peer {p} fetch failed: {e}")
    return out
