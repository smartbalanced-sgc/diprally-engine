"""FMP + yfinance wrappers. W0: byte-equivalent ports from seed v1.

W2 critical-fixes additions:
  - URL apikey redaction in all error logs (security: prevent key leak)
  - FetchError typed exception (resilience: graceful caller handling)
  - HTTP errors wrapped, not raised raw (resilience: don't crash pipeline)
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta

import pandas as pd
import requests

from src.config import FMP_BASE


_APIKEY_RE = re.compile(r"apikey=[^&\s'\"]*", re.IGNORECASE)


def _redact(text) -> str:
    """Scrub apikey=... query params from any string. Used in every error
    log path that could include a URL — prevents API key leakage when error
    messages are pasted into chat / GitHub issues / log aggregators.
    """
    return _APIKEY_RE.sub("apikey=***REDACTED***", str(text))


class FetchError(Exception):
    """Typed exception for data-fetch failures.

    Carries (ticker, source, status, reason) so the caller can decide:
      - single-ticker CLI: exit non-zero with a clean message
      - batch orchestrator (W5): skip ticker with WARNING, continue with the
        remaining 16

    All `reason` strings are pre-redacted via _redact() so apikey never leaks.
    """
    def __init__(self, ticker: str, source: str, status, reason: str):
        self.ticker = ticker
        self.source = source
        self.status = status
        self.reason = _redact(reason)
        status_part = f" [{status}]" if status is not None else ""
        super().__init__(f"{source} fetch failed for {ticker}{status_part}: {self.reason}")


def _fetch_history_fmp(ticker: str, api_key: str, lookback_days: int) -> pd.DataFrame:
    """FMP-only history fetch. Raises FetchError on any failure. Internal."""
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    url = f"{FMP_BASE}/historical-price-eod/full"
    params = {"symbol": ticker, "from": start, "to": end, "apikey": api_key}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else None
        hint = ""
        if status == 402:
            hint = " (ticker not in FMP plan tier)"
        elif status == 429:
            hint = " (FMP rate limit — back off)"
        raise FetchError(ticker, "fmp", status, f"HTTP error{hint}") from None
    except requests.exceptions.Timeout:
        raise FetchError(ticker, "fmp", None, "request timeout") from None
    except requests.exceptions.RequestException as e:
        raise FetchError(ticker, "fmp", None, f"network error: {e}") from None

    try:
        data = r.json()
    except ValueError as e:
        raise FetchError(ticker, "fmp", r.status_code, f"non-JSON response: {e}") from None

    if not isinstance(data, list) or not data:
        raise FetchError(ticker, "fmp", r.status_code, "empty response — ticker may not exist")
    df = pd.DataFrame(data).rename(columns={"date": "Date", "close": "Close"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df.attrs["data_source"] = "fmp"
    return df


def _fetch_history_yfinance(ticker: str, lookback_days: int) -> pd.DataFrame:
    """yfinance history fetch. Raises FetchError on any failure. Internal.

    Used as the fallback provider when FMP fails (D-W2-14). yfinance covers
    a wider universe than FMP Starter (e.g. handles MOG.A dot-form fine,
    handles all NYSE/NASDAQ names including microcaps and class shares).
    Free; no apikey; rate-limited softly by Yahoo but typically generous.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise FetchError(ticker, "yfinance", None, "yfinance not installed") from None

    try:
        tk = yf.Ticker(ticker)
        # Calendar days; yfinance period strings: "1y", "2y", "5y"; or
        # explicit start/end. Use start/end to match FMP semantics.
        start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        end = datetime.now().strftime("%Y-%m-%d")
        df = tk.history(start=start, end=end, auto_adjust=False)
    except Exception as e:
        raise FetchError(ticker, "yfinance", None, f"history fetch error: {e}") from None

    if df is None or df.empty:
        raise FetchError(ticker, "yfinance", None, "empty response")

    # Normalize to FMP-compatible shape: Date + Close columns.
    df = df.reset_index().rename(columns={"index": "Date"})
    if "Date" not in df.columns:
        # Some yfinance versions name the index column 'Date' already
        df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    if "Close" not in df.columns:
        raise FetchError(ticker, "yfinance", None, "no Close column in response")
    df = df.sort_values("Date").reset_index(drop=True)
    df.attrs["data_source"] = "yfinance"
    return df


def fetch_history(ticker: str, api_key: str, lookback_days: int) -> pd.DataFrame:
    """Fetch OHLC history with FMP-primary, yfinance-fallback resilience.

    D-W2-14: closes the single-point-of-failure that previously meant one bad
    FMP response (402/404/429/5xx/timeout) killed the entire pipeline. Now:

      1. Try FMP first (cheap, higher-quality OHLC, lower latency)
      2. If FMP raises FetchError, log the reason and retry on yfinance
      3. If yfinance also fails, raise a combined FetchError with both reasons

    Per-provider symbol translation (when needed) lives in the registry
    (sacred #17). Today's universe uses the same dash form on both providers,
    so caller passes one canonical ticker and both providers see it.

    Return df.attrs['data_source'] is 'fmp' or 'yfinance' so downstream
    consumers (CSV row, reporter) can log which provider supplied the data.
    """
    # Per-provider symbol resolution (registry handles divergences if any).
    # Imported lazily to avoid circular import (registry imports config).
    try:
        from src.registry import provider_symbol
        fmp_sym = provider_symbol(ticker, "fmp")
        yf_sym = provider_symbol(ticker, "yfinance")
    except Exception:
        # Registry lookup may fail for unknown tickers — use canonical.
        fmp_sym = yf_sym = ticker

    # Primary: FMP
    try:
        return _fetch_history_fmp(fmp_sym, api_key, lookback_days)
    except FetchError as fmp_err:
        print(f"   ⚠ FMP fetch failed for {ticker}: {fmp_err.reason} — falling back to yfinance")
        # Fallback: yfinance
        try:
            return _fetch_history_yfinance(yf_sym, lookback_days)
        except FetchError as yf_err:
            # Both providers failed. Raise combined error so caller sees
            # both reasons (helps debugging — was it a network outage,
            # a tier issue, or a genuinely-unknown ticker?).
            combined = (f"FMP failed ({fmp_err.reason}); "
                        f"yfinance fallback also failed ({yf_err.reason})")
            raise FetchError(ticker, "all-providers", None, combined) from None


def _fmp_get(endpoint, api_key, params=None):
    """Best-effort FMP fetch for SUPPLEMENTARY endpoints (analyst targets, sector
    perf, news, etc.). Returns None on failure with a redacted-URL warning log.

    Use fetch_history (which raises FetchError) for endpoints where missing
    data should abort the whole pipeline. Use _fmp_get for endpoints where
    a missing fetch should degrade-gracefully (signal becomes _none_signal).

    PR #89: rate-limit-aware retry. If FMP returns HTTP 429 (rate-limited),
    wait briefly and retry once. Empirically Starter plan tolerates ~3 req/sec
    sustained at parallel-4; this retry catches occasional bursts above that.
    Single retry only — if it persists, fall back to degrade-gracefully None.
    """
    import time as _time
    p = {"apikey": api_key}
    if params:
        p.update(params)
    for attempt in (1, 2):
        try:
            r = requests.get(f"{FMP_BASE}/{endpoint}", params=p, timeout=15)
            if r.status_code == 429 and attempt == 1:
                # Rate-limited. Honour Retry-After if present; otherwise 2s.
                wait_s = float(r.headers.get("Retry-After", 2))
                wait_s = min(max(wait_s, 1.0), 5.0)  # clamp 1-5s
                print(f"   ℹ FMP {endpoint} rate-limited (HTTP 429); "
                      f"waiting {wait_s:.1f}s and retrying once.")
                _time.sleep(wait_s)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == 1:
                # First-try failures (network, transient) get a single retry
                # too, with a short jitter to avoid synchronized thundering-herd
                # when parallel-4 subprocesses all see the same transient.
                _time.sleep(0.5)
                continue
            print(f"   WARNING: FMP {endpoint} failed after retry: {_redact(e)}")
            return None
    return None


def fetch_live_quote(ticker, api_key):
    """Real-time quote from FMP /stable/quote.

    PR #85 — fixes the data-layer rot exposed on 2026-05-26: the engine
    had been setting `spot = history_df["Close"].iloc[-1]`, i.e. the
    last completed daily bar. FMP's daily-bar endpoint lags 1-2 hours
    after market close, so a Tue 16:50 ET cycle saw Friday's close for
    SNDK / MU / AMAT — missing the +7.5% / +19.3% / +5.3% Tue rallies
    entirely. Every BUY recommendation that cycle was computed against
    stale spots.

    `/stable/quote` returns CURRENT intraday price during market hours,
    or the most recent session's close after-hours. Plan-tier-confirmed
    (Starter) endpoint per FMP support 2026-05-27.

    Returns dict with keys:
        price            — current intraday or last session's close
        previous_close   — prior session's close
        day_low, day_high
        open
        volume
        change_pct       — percent change vs previous close
        timestamp        — Unix epoch (recent if quote is live)
        symbol, exchange
    or None on HTTP error, empty response, or unparseable payload.
    Callers MUST fall back gracefully (e.g. to daily-bar last close)
    rather than abort, since live quotes can be flaky for newly-listed
    tickers or during exchange data outages.
    """
    data = _fmp_get("quote", api_key, {"symbol": ticker})
    if not data or not isinstance(data, list) or not data:
        return None
    d = data[0]
    try:
        price = float(d.get("price"))
    except (TypeError, ValueError):
        return None
    if price <= 0:
        return None
    return {
        "symbol": d.get("symbol", ticker),
        "exchange": d.get("exchange"),
        "price": price,
        "previous_close": (
            float(d["previousClose"])
            if d.get("previousClose") is not None else None
        ),
        "day_low": (
            float(d["dayLow"]) if d.get("dayLow") is not None else None
        ),
        "day_high": (
            float(d["dayHigh"]) if d.get("dayHigh") is not None else None
        ),
        "open": (
            float(d["open"]) if d.get("open") is not None else None
        ),
        "volume": d.get("volume"),
        "change_pct": d.get("changePercentage"),
        "timestamp": d.get("timestamp"),
    }


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


def fetch_grades_history(ticker, api_key, limit=200, lookback_days=120):
    """W6 PR #35 — analyst upgrade/downgrade history.

    FMP's stable-API endpoint for individual grade-change events is
    `/stable/grades?symbol=X` (PR #43 — confirmed with FMP support).
    Each row has date, gradingCompany, previousGrade, newGrade, action
    ("upgrade" / "downgrade" / "maintain"). The signal filters to
    upgrade/downgrade only.

    PR #44 update: FMP's `limit` param is IGNORED on this endpoint
    (returns all 810 rows for INTC regardless). We truncate client-side
    by date — keep only rows within `lookback_days` of today (default
    120d, comfortably wider than the signal's 90d filter window). This
    drops ~95% of bytes per fetch while preserving signal correctness.
    Also caps at `limit` rows as a hard safety net for tickers with
    extreme coverage history.
    """
    data = _fmp_get("grades", api_key,
                     {"symbol": ticker})
    if not data or not isinstance(data, list):
        return []
    # Client-side date truncation — FMP `limit` is ignored on stable grades.
    today = datetime.now().date()
    cutoff = today - timedelta(days=lookback_days)
    filtered = []
    for row in data:
        if not isinstance(row, dict):
            continue
        date_str = row.get("date") or row.get("publishedDate") or ""
        try:
            row_date = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
        if row_date < cutoff:
            continue
        # Normalize field name — signal reads publishedDate but FMP stable
        # uses 'date'. Keep both so the signal works either way.
        if "publishedDate" not in row and "date" in row:
            row = dict(row)
            row["publishedDate"] = row["date"]
        filtered.append(row)
        if len(filtered) >= limit:
            break
    return filtered


def fetch_fundamentals(ticker, api_key, market_cap=None):
    """W6 PR #34 — TTM FCF + leverage + margin trend.

    Pulls from two FMP endpoints:
      * key-metrics-ttm  → TTM FCF, EBITDA, net debt (per-share scaled)
      * income-statement?period=quarter  → last 8 quarters for the
                                            margin-trend sub-component

    Returns a dict {fcf_yield, net_debt_to_ebitda, op_margin_trend,
    n_components_available}. Each numeric field is None when not
    computable (pre-revenue names, missing endpoint, etc.); the signal
    function treats None as "this sub-component unavailable" rather
    than as zero.
    """
    out = {
        "fcf_yield": None,
        "net_debt_to_ebitda": None,
        "op_margin_trend": None,
        "n_components_available": 0,
    }

    # TTM key metrics
    ttm = _fmp_get("key-metrics-ttm", api_key, {"symbol": ticker})
    if ttm and isinstance(ttm, list) and ttm:
        row = ttm[0]
        # FCF yield: FMP exposes freeCashFlowYieldTTM directly (decimal).
        fcfy = row.get("freeCashFlowYieldTTM")
        if fcfy is not None:
            try:
                out["fcf_yield"] = float(fcfy)
                out["n_components_available"] += 1
            except (TypeError, ValueError):
                pass
        # Net debt / EBITDA: FMP exposes netDebtToEBITDATTM. Only valid
        # when EBITDA > 0 (negative-EBITDA names get None — leverage
        # ratio is meaningless against negative cash flow). We infer
        # EBITDA sign from evToEBITDATTM: EV is generally positive, so
        # evToEBITDATTM > 0 ⇒ EBITDA > 0. (PR #43: prior code had a
        # camelCase typo "evToEbitdaTTM" — actual field is uppercase
        # "evToEBITDATTM" — and stable doesn't expose ebitdaTTM directly,
        # so the sub-component was silently returning None on every
        # ticker since PR #34 shipped.)
        nd_ebitda = row.get("netDebtToEBITDATTM")
        ev_to_ebitda = row.get("evToEBITDATTM")
        if (nd_ebitda is not None and ev_to_ebitda is not None
                and float(ev_to_ebitda) > 0):
            try:
                out["net_debt_to_ebitda"] = float(nd_ebitda)
                out["n_components_available"] += 1
            except (TypeError, ValueError):
                pass

    # Quarterly income statement for margin-trend. PR #37 hotfix:
    # stable API uses ?symbol=X (query-param style), not path-style.
    inc = _fmp_get("income-statement", api_key,
                    {"symbol": ticker, "period": "quarter", "limit": 8})
    if inc and isinstance(inc, list) and len(inc) >= 8:
        try:
            # FMP rows are newest-first.
            recent_4q = inc[:4]
            prior_4q = inc[4:8]
            def _avg_margin(rows):
                margins = []
                for r in rows:
                    rev = float(r.get("revenue") or 0.0)
                    op_inc = float(r.get("operatingIncome") or 0.0)
                    if rev > 0:
                        margins.append(op_inc / rev)
                return sum(margins) / len(margins) if margins else None
            recent_avg = _avg_margin(recent_4q)
            prior_avg = _avg_margin(prior_4q)
            if recent_avg is not None and prior_avg is not None:
                out["op_margin_trend"] = recent_avg - prior_avg
                out["n_components_available"] += 1
        except (TypeError, ValueError, KeyError):
            pass

    return out


def fetch_sector_perf(sector, api_key, days=None, exchange_filter="NASDAQ"):
    """FMP historical-sector-performance — filtered to one exchange, last-N unique dates.

    `days` defaults to config sector_perf.default_lookback_days (sacred #17,
    D-W2-8). Callers can override via the argument.
    """
    if days is None:
        from src.config import SECTOR_PERF_DEFAULT_LOOKBACK_DAYS
        days = SECTOR_PERF_DEFAULT_LOOKBACK_DAYS
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
    """FMP VIX + SPY for risk-on/risk-off. Thresholds in config/diprally.yaml
    under macro_regime (sacred #17, D-W2-8)."""
    from src.config import (
        SPY_RISK_OFF_THRESHOLD,
        SPY_RISK_ON_THRESHOLD,
        VIX_DEFAULT_FALLBACK,
        VIX_RISK_OFF_THRESHOLD,
        VIX_RISK_ON_THRESHOLD,
    )
    vix_data = _fmp_get("quote", api_key, {"symbol": "^VIX"})
    spy_data = _fmp_get("quote", api_key, {"symbol": "SPY"})
    vix = VIX_DEFAULT_FALLBACK
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
    if vix > VIX_RISK_OFF_THRESHOLD or spy_trend < SPY_RISK_OFF_THRESHOLD:
        regime = "risk_off"
    elif vix < VIX_RISK_ON_THRESHOLD and spy_trend > SPY_RISK_ON_THRESHOLD:
        regime = "risk_on"
    else:
        regime = "neutral"
    return {"vix": vix, "spy_trend": spy_trend, "regime": regime}


def fetch_options_iv(ticker, target_dte_days=None):
    """yfinance options chain → ATM straddle IV at ~target_dte_days expiry.

    Liquidity-gated: only returns IV if option chain is liquid enough.
    Thresholds in config/diprally.yaml under options_iv (sacred #17, D-W2-8).
    """
    from src.config import (
        OPTIONS_IV_DEFAULT_TARGET_DTE_DAYS,
        OPTIONS_IV_DTE_WINDOW_MAX_MULTIPLIER,
        OPTIONS_IV_DTE_WINDOW_MIN,
        OPTIONS_IV_LIQUIDITY_MAX_SPREAD,
    )
    if target_dte_days is None:
        target_dte_days = OPTIONS_IV_DEFAULT_TARGET_DTE_DAYS
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
        dte_max = target_dte_days * OPTIONS_IV_DTE_WINDOW_MAX_MULTIPLIER
        for ex_str in expiries:
            try:
                ex_date = datetime.strptime(ex_str, "%Y-%m-%d").date()
                dte = (ex_date - today).days
                if OPTIONS_IV_DTE_WINDOW_MIN <= dte <= dte_max:
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
        is_liquid = avg_spread < OPTIONS_IV_LIQUIDITY_MAX_SPREAD
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
        print(f"   WARNING: yfinance options IV fetch failed: {_redact(e)}")
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
        print(f"   WARNING: yfinance short-interest fetch failed: {_redact(e)}")
    return None


def fetch_peer_history(peers, api_key, lookback_days=60):
    """Fetch closing-price history for peer tickers."""
    out = {}
    for p in peers:
        try:
            df = fetch_history(p, api_key, lookback_days=lookback_days)
            if df is not None and not df.empty:
                out[p] = df
        except FetchError as e:
            # FetchError already has _redact applied to its reason field
            print(f"   WARNING: peer {p} fetch failed: {e}")
        except Exception as e:
            print(f"   WARNING: peer {p} fetch failed: {_redact(e)}")
    return out
