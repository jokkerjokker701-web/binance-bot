from __future__ import annotations

import hashlib
import hmac
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlencode

import requests


# -------------------- Errors --------------------

@dataclass
class BinanceHTTPError(Exception):
    status_code: int
    url: str
    body: str

    def __str__(self) -> str:
        return f"Binance HTTP {self.status_code} for {self.url}: {self.body}"


# -------------------- Env / Config --------------------

def _env_first(*keys: str, default: Optional[str] = None) -> Optional[str]:
    for k in keys:
        v = os.getenv(k)
        if v is not None and str(v).strip() != "":
            return v.strip()
    return default


def get_api_key() -> str:
    v = _env_first("BINANCE_API_KEY", "API_KEY", "BINANCE_KEY")
    return v or ""


def get_api_secret() -> str:
    v = _env_first("BINANCE_API_SECRET", "API_SECRET", "BINANCE_SECRET")
    return v or ""


def is_testnet() -> bool:
    v = _env_first("BINANCE_TESTNET", "TESTNET", "USE_TESTNET", default="0")
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def futures_base_url() -> str:
    """
    Priority:
      1) BINANCE_FUTURES_BASE_URL
      2) FUTURES_URL / FUTURES_BASE_URL
      3) from BINANCE_TESTNET flag
    """
    u = _env_first("BINANCE_FUTURES_BASE_URL", "FUTURES_URL", "FUTURES_BASE_URL")
    if u:
        return u.rstrip("/")
    return ("https://testnet.binancefuture.com" if is_testnet() else "https://fapi.binance.com").rstrip("/")


# -------------------- Low-level request helpers --------------------

_SESSION = requests.Session()
_SESSION.headers.update({"Content-Type": "application/x-www-form-urlencoded"})


def _sign_query(query_string: str, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), query_string.encode("utf-8"), hashlib.sha256).hexdigest()


def _request(
    method: str,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    signed: bool = False,
    timeout: int = 20,
) -> Dict[str, Any]:
    base = futures_base_url()
    url = f"{base}{path}"

    params = dict(params or {})

    headers = {}
    api_key = get_api_key()
    api_secret = get_api_secret()

    if signed:
        if not api_key or not api_secret:
            raise BinanceHTTPError(401, url, '{"msg":"Missing BINANCE_API_KEY / BINANCE_API_SECRET"}')

        # Binance requires timestamp in ms
        params["timestamp"] = int(time.time() * 1000)
        # optional but helps reduce timestamp errors on some networks
        params.setdefault("recvWindow", 5000)

        qs = urlencode(params, doseq=True)
        sig = _sign_query(qs, api_secret)
        qs = f"{qs}&signature={sig}"
        headers["X-MBX-APIKEY"] = api_key
    else:
        qs = urlencode(params, doseq=True) if params else ""

    try:
        if method.upper() in ("GET", "DELETE"):
            full_url = f"{url}?{qs}" if qs else url
            r = _SESSION.request(method.upper(), full_url, headers=headers, timeout=timeout)
        else:
            # POST/PUT: send params in querystring (Binance accepts form too)
            full_url = f"{url}?{qs}" if qs else url
            r = _SESSION.request(method.upper(), full_url, headers=headers, timeout=timeout)
    except Exception as e:
        raise BinanceHTTPError(599, url, f'{{"msg":"network_error","err":"{e}"}}')

    text = r.text or ""
    if not r.ok:
        # print helpful log
        print(f"[BINANCE_ERR] {r.status_code} {url} {text}", flush=True)
        raise BinanceHTTPError(r.status_code, url, text)

    if text.strip() == "":
        return {}
    try:
        return r.json()
    except Exception:
        # Some endpoints may return plain text
        return {"raw": text}


# -------------------- Public endpoints --------------------

def ping() -> Dict[str, Any]:
    return _request("GET", "/fapi/v1/ping", signed=False)


def server_time() -> Dict[str, Any]:
    return _request("GET", "/fapi/v1/time", signed=False)


def exchange_info() -> Dict[str, Any]:
    return _request("GET", "/fapi/v1/exchangeInfo", signed=False)


def ticker_price(symbol: str) -> float:
    data = _request("GET", "/fapi/v1/ticker/price", params={"symbol": symbol}, signed=False)
    return float(data.get("price", 0.0))


# -------------------- Signed/account endpoints --------------------

def account() -> Dict[str, Any]:
    return _request("GET", "/fapi/v2/account", signed=True)


def position_risk(symbol: Optional[str] = None) -> Dict[str, Any]:
    params = {"symbol": symbol} if symbol else None
    return _request("GET", "/fapi/v2/positionRisk", params=params, signed=True)


def open_orders(symbol: Optional[str] = None) -> Dict[str, Any]:
    params = {"symbol": symbol} if symbol else None
    return _request("GET", "/fapi/v1/openOrders", params=params, signed=True)


def get_order(symbol: str, order_id: int) -> Dict[str, Any]:
    return _request("GET", "/fapi/v1/order", params={"symbol": symbol, "orderId": int(order_id)}, signed=True)


def cancel_order(symbol: str, order_id: int) -> Dict[str, Any]:
    return _request("DELETE", "/fapi/v1/order", params={"symbol": symbol, "orderId": int(order_id)}, signed=True)


def set_leverage(symbol: str, leverage: int) -> Dict[str, Any]:
    leverage = int(leverage)
    leverage = max(1, min(leverage, 125))
    return _request("POST", "/fapi/v1/leverage", params={"symbol": symbol, "leverage": leverage}, signed=True)


def set_margin_type(symbol: str, margin_type: str) -> Dict[str, Any]:
    """
    margin_type: ISOLATED or CROSSED
    """
    mt = str(margin_type).upper().strip()
    if mt not in ("ISOLATED", "CROSSED"):
        raise ValueError("margin_type must be ISOLATED or CROSSED")
    try:
        return _request("POST", "/fapi/v1/marginType", params={"symbol": symbol, "marginType": mt}, signed=True)
    except BinanceHTTPError as e:
        # -4046 "No need to change margin type."
        if '"code":-4046' in str(e) or "No need to change margin type" in str(e):
            return {"code": 200, "msg": "already_set"}
        raise


# -------------------- Position mode (Hedge / One-way) --------------------

def get_position_mode() -> Dict[str, Any]:
    """
    Returns: {"dualSidePosition": True/False}
    True  => HEDGE mode (LONG/SHORT separate)
    False => ONEWAY mode (BOTH)
    """
    return _request("GET", "/fapi/v1/positionSide/dual", signed=True)


def set_position_mode(dual: bool) -> Dict[str, Any]:
    """
    dual=True  -> enable Hedge mode
    dual=False -> enable One-way mode
    """
    try:
        return _request(
            "POST",
            "/fapi/v1/positionSide/dual",
            params={"dualSidePosition": "true" if dual else "false"},
            signed=True,
        )
    except BinanceHTTPError as e:
        # -4059 "No need to change position side."
        if '"code":-4059' in str(e) or "No need to change position side" in str(e):
            return {"code": 200, "msg": "already_set"}
        raise


def _infer_position_side_for_hedge(side: str) -> str:
    s = side.upper().strip()
    # Hedge mode:
    # BUY  -> LONG
    # SELL -> SHORT
    return "LONG" if s == "BUY" else "SHORT"


def _resolve_position_side(side: str, position_side: Optional[str]) -> Tuple[bool, Optional[str]]:
    """
    Returns: (is_hedge, positionSide or None)
    - If one-way => return (False, None) and DO NOT send positionSide at all
    - If hedge  => return (True, 'LONG'/'SHORT')
    """
    mode = get_position_mode()
    hedge = bool(mode.get("dualSidePosition", False))

    if not hedge:
        # ONEWAY: do NOT send positionSide (Binance can error if you send it wrong)
        return False, None

    # HEDGE
    if position_side:
        ps = position_side.upper().strip()
        if ps not in ("LONG", "SHORT"):
            raise ValueError("position_side must be LONG or SHORT in hedge mode")
        return True, ps

    # auto infer
    return True, _infer_position_side_for_hedge(side)


# -------------------- Orders --------------------

def test_order(
    symbol: str,
    side: str,
    quantity: float,
    *,
    order_type: str = "MARKET",
    position_side: Optional[str] = None,
    reduce_only: bool = False,
    time_in_force: Optional[str] = None,
    price: Optional[float] = None,
    stop_price: Optional[float] = None,
    working_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Binance "order/test" endpoint.
    Validates order parameters WITHOUT actually placing an order.
    """
    params: Dict[str, Any] = {
        "symbol": symbol,
        "side": side.upper(),
        "type": order_type.upper(),
        "quantity": f"{float(quantity):.6f}",
    }

    hedge, ps = _resolve_position_side(side, position_side)
    if hedge and ps:
        params["positionSide"] = ps

    if reduce_only:
        params["reduceOnly"] = "true"

    if time_in_force:
        params["timeInForce"] = time_in_force

    if price is not None:
        params["price"] = f"{float(price):.2f}"

    if stop_price is not None:
        params["stopPrice"] = f"{float(stop_price):.2f}"

    if working_type:
        params["workingType"] = working_type

    return _request("POST", "/fapi/v1/order/test", params=params, signed=True)


def new_order(
    symbol: str,
    side: str,
    quantity: float,
    position_side: Optional[str] = None,
    *,
    order_type: str = "MARKET",
    reduce_only: bool = False,
    close_position: bool = False,
    time_in_force: Optional[str] = None,
    price: Optional[float] = None,
    stop_price: Optional[float] = None,
    working_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Places a futures order.

    ✅ Key fix: It auto-detects your account position mode:
      - One-way mode -> DOES NOT send positionSide
      - Hedge mode   -> sends positionSide=LONG/SHORT automatically (or uses provided)

    This prevents:
      -4061 Order's position side does not match user's setting
    """
    side_u = side.upper().strip()
    if side_u not in ("BUY", "SELL"):
        raise ValueError("side must be BUY or SELL")

    params: Dict[str, Any] = {
        "symbol": symbol,
        "side": side_u,
        "type": order_type.upper(),
    }

    # quantity optional if close_position=True
    if not close_position:
        params["quantity"] = f"{float(quantity):.6f}"

    hedge, ps = _resolve_position_side(side_u, position_side)
    if hedge and ps:
        params["positionSide"] = ps

    if reduce_only:
        params["reduceOnly"] = "true"

    if close_position:
        params["closePosition"] = "true"

    if time_in_force:
        params["timeInForce"] = time_in_force

    if price is not None:
        params["price"] = f"{float(price):.2f}"

    if stop_price is not None:
        params["stopPrice"] = f"{float(stop_price):.2f}"

    if working_type:
        params["workingType"] = working_type

    return _request("POST", "/fapi/v1/order", params=params, signed=True)