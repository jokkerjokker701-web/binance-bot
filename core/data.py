import time
import httpx

# Futures public endpoint (real data). Demo endpoint ba'zida beqaror bo'lishi mumkin.
PUBLIC_FAPI = "https://fapi.binance.com"

# global client: connection pooling + kamroq handshake
_client = httpx.Client(
    timeout=httpx.Timeout(10.0, connect=10.0),
    headers={"User-Agent": "binance-bot/1.0"},
)

def fetch_klines(symbol: str, interval: str, limit: int, retries: int = 6):
    """
    Robust kline fetcher:
    - retries with exponential backoff
    - handles WinError 10054 / TLS handshake / transient issues
    """
    last_err = None
    url = f"{PUBLIC_FAPI}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    for attempt in range(retries):
        try:
            r = _client.get(url, params=params)
            r.raise_for_status()
            data = r.json()

            # data -> DataFrame formatini sizning projectga mos qaytaramiz:
            # Sizda avval df = fetch_klines(...) ishlayapti, demak bu yerda
            # pandas DataFrame qaytishi kerak. Agar sizda core/data.py allaqachon
            # pandas bilan ishlayotgan bo'lsa, pastdagi blokni moslab qo'ying.
            import pandas as pd

            # Binance kline format:
            # 0 openTime,1 open,2 high,3 low,4 close,5 volume,6 closeTime,...
            rows = []
            for k in data:
                rows.append({
                    "open_ts": int(k[0]),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "close_ts": int(k[6]),
                })

            return pd.DataFrame(rows)

        except (httpx.TimeoutException, httpx.NetworkError, httpx.ProtocolError, httpx.HTTPError) as e:
            last_err = e
            # backoff: 1s,2s,4s,8s...
            sleep_s = min(30, 2 ** attempt)
            time.sleep(sleep_s)

        except Exception as e:
            last_err = e
            time.sleep(2)

    raise RuntimeError(f"fetch_klines failed after {retries} retries: {last_err}")