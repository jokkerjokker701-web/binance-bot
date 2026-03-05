from __future__ import annotations

import hashlib
import hmac
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Literal
from urllib.parse import urlencode

import requests


# -------------------------
# Config (env)
# -------------------------
DEFAULT_BASE_URL = os.getenv("BINANCE_FUTURES_URL", "https://testnet.binancefuture.com")

API_KEY = os.getenv("BINANCE_API_KEY", "") or os.getenv("BINANCE_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "") or os.getenv("BINANCE_SECRET", "")

SESSION = requests.Session()
if API_KEY:
    SESSION.headers.update({"X-MBX-APIKEY": API_KEY})


# -------------------------
# Errors
# -------------------------
class BinanceHTTPError(RuntimeError):
    def __init__(self, status: int, url: str, body: str):
        super().__init__(f"Binance HTTP {status} for {url}: {body}")
        self.status = status
        self.url = url
        self.body = body


def _now_ms() -> int:
    return int(time.time() * 1000)


def _sign(query: str) -> str:
    if not API_SECRET:
        raise RuntimeError("BINANCE_API_SECRET not set")
    return hmac.new(API_SECRET.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()


def _request(
    method: str,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    signed: bool = False,
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = 20,
) -> Dict[str, Any]:
    params = dict(params or {})

    if signed:
        params["timestamp"] = _now_ms()
        # recvWindow optional, but helpful
        params.setdefault("recvWindow", 5000)

        query = urlencode(params, doseq=True)
        params["signature"] = _sign(query)

    url = base_url.rstrip("/") + path

    try:
        if method.upper() == "GET":
            r = SESSION.get(url, params=params, timeout=timeout)
        elif method.upper() == "POST":
            r = SESSION.post(url, params=params, timeout=timeout)
        elif method.upper() == "DELETE":
            r = SESSION.delete(url, params=params, timeout=timeout)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
    except Exception as e:
        raise RuntimeError(f"HTTP request failed: {e}") from e

    if not r.ok:
        body = r.text
        # print helpful line for debugging
        print(f"[BINANCE_ERR] {r.status_code} {r.url} {body}")
        raise BinanceHTTPError(r.status_code, r.url, body)

    # Binance sometimes returns empty body on success
    if not r.text.strip():
        return {}
    return r.json()


# -------------------------
# Basic endpoints
# -------------------------
def ping() -> Dict[str, Any]:
    return _request("GET", "/fapi/v1/ping", signed=False)


def server_time() -> Dict[str, Any]:
    return _request("GET", "/fapi/v1/time", signed=False)


def exchange_info() -> Dict[str, Any]:
    return _request("GET", "/fapi/v1/exchangeInfo", signed=False)


def account() -> Dict[str, Any]:
    return _request("GET", "/fapi/v2/account", signed=True)


def open_orders(symbol: Optional[str] = None) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if symbol:
        params["symbol"] = symbol
    return _request("GET", "/fapi/v1/openOrders", params=params, signed=True)


def cancel_all_orders(symbol: str) -> Dict[str, Any]:
    return _request("DELETE", "/fapi/v1/allOpenOrders", params={"symbol": symbol}, signed=True)


def get_order(symbol: str, order_id: int) -> Dict[str, Any]:
    return _request("GET", "/fapi/v1/order", params={"symbol": symbol, "orderId": int(order_id)}, signed=True)


# -------------------------
# Position Mode (Hedge/One-way)
# -------------------------
def get_position_mode() -> Dict[str, Any]:
    """
    Returns:
      {"dualSidePosition": true/false}
    """
    return _request("GET", "/fapi/v1/positionSide/dual", signed=True)


def set_position_mode(dual: bool) -> Dict[str, Any]:
    """
    dual=True  -> Hedge Mode (LONG/SHORT)
    dual=False -> One-way Mode (BOTH)
    Notes:
      If already set, Binance returns code -4059.
      Biz buni xato deb emas, "OK" deb qabul qilamiz.
    """
    try:
        return _request(
            "POST",
            "/fapi/v1/positionSide/dual",
            params={"dualSidePosition": "true" if dual else "false"},
            signed=True,
        )
    except BinanceHTTPError as e:
        # -4059: No need to change position side.
        if '"code":-4059' in str(e) or "No need to change position side" in str(e):
            return {"code": 200, "msg": "already_set"}
        raise


# -------------------------
# Leverage/Margin helpers (optional)
# -------------------------
def set_leverage(symbol: str, leverage: int) -> Dict[str, Any]:
    leverage = int(leverage)
    return _request("POST", "/fapi/v1/leverage", params={"symbol": symbol, "leverage": leverage}, signed=True)


def set_margin_type(symbol: str, margin_type: Literal["ISOLATED", "CROSSED"]) -> Dict[str, Any]:
    return _request("POST", "/fapi/v1/marginType", params={"symbol": symbol, "marginType": margin_type}, signed=True)


# -------------------------
# ORDER (AUTO positionSide fix)
# -------------------------
def _normalize_side(side: str) -> str:
    s = side.upper().strip()
    if s not in ("BUY", "SELL"):
        raise ValueError("side must be BUY or SELL")
    return s


def _auto_position_side_for_hedge(side: str) -> str:
    # Binance hedge: BUY->LONG, SELL->SHORT (default for opening)
    return "LONG" if side == "BUY" else "SHORT"


def new_order(
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
    close_position: bool = False,
    working_type: Optional[str] = None,  # CONTRACT_PRICE / MARK_PRICE
) -> Dict[str, Any]:
    """
    ✅ Eng muhim fix:
      - account hedge bo‘lsa -> positionSide LONG/SHORT bo‘ladi (AUTO)
      - account one-way bo‘lsa -> positionSide yuborilmaydi

    position_side ni qo'lda berishingiz shart emas.
    Agar bersangiz ham, mode bilan mos bo‘lmasa avtomatik to‘g‘rilanadi.
    """
    symbol = symbol.upper().strip()
    side_n = _normalize_side(side)

    q = float(quantity)
    if q <= 0:
        raise ValueError("quantity must be > 0")

    # detect mode
    mode = get_position_mode()
    dual = bool(mode.get("dualSidePosition", False))

    # Build params
    params: Dict[str, Any] = {
        "symbol": symbol,
        "side": side_n,
        "type": order_type.upper().strip(),
        "quantity": f"{q:.6f}".rstrip("0").rstrip("."),
    }

    # AUTO: positionSide
    if dual:
        # Hedge mode requires LONG/SHORT
        ps = (position_side or "").upper().strip()
        if ps not in ("LONG", "SHORT"):
            ps = _auto_position_side_for_hedge(side_n)
        params["positionSide"] = ps
    else:
        # One-way mode: DO NOT send positionSide (or BOTH).
        # safest is: not sending at all
        pass

    if reduce_only:
        params["reduceOnly"] = "true"
    if close_position:
        params["closePosition"] = "true"

    if time_in_force:
        params["timeInForce"] = time_in_force
    if price is not None:
        params["price"] = str(price)
    if stop_price is not None:
        params["stopPrice"] = str(stop_price)
    if working_type:
        params["workingType"] = working_type

    return _request("POST", "/fapi/v1/order", params=params, signed=True)