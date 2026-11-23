"""Thin Kalshi Trade API v2 client.

Auth follows Kalshi's official SDK (kalshi-python-sync auth.py): sign
`timestamp_ms + METHOD + /trade-api/v2/path` (no query string) with RSA-PSS/SHA-256.
Endpoint paths mirror the official OpenAPI spec (v3.36+).
"""
from __future__ import annotations

import base64
import time
from urllib.parse import urlparse

import requests

from . import config


class KalshiError(RuntimeError):
    pass


class Kalshi:
    def __init__(self, base_url: str, key_id: str = "", private_key_path: str = ""):
        self.base = base_url.rstrip("/")
        self.base_path = urlparse(self.base).path
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "kalshi-nba-combo-bot/0.1"
        self.key_id = key_id
        self._key = None
        if key_id and private_key_path:
            from cryptography.hazmat.primitives import serialization
            with open(private_key_path, "rb") as f:
                self._key = serialization.load_pem_private_key(f.read(), password=None)

    @property
    def authed(self) -> bool:
        return self._key is not None

    # ---------------------------------------------------------------- transport
    def _auth_headers(self, method: str, path: str) -> dict:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding, ed25519
        ts = str(int(time.time() * 1000))
        msg = (ts + method.upper() + self.base_path + path).encode()
        if isinstance(self._key, ed25519.Ed25519PrivateKey):
            sig = self._key.sign(msg)
        else:
            sig = self._key.sign(
                msg,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
                hashes.SHA256(),
            )
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        }

    def request(self, method: str, path: str, params=None, json=None, auth=False, retries=4):
        if auth and not self.authed:
            raise KalshiError("This call needs KALSHI_API_KEY_ID + KALSHI_PRIVATE_KEY_PATH")
        for attempt in range(retries + 40):
            headers = self._auth_headers(method, path) if (auth or self.authed) else {}
            r = self.s.request(method, self.base + path, params=params, json=json, headers=headers, timeout=30)
            if r.status_code == 429 and attempt < retries + 36:
                # token bucket: the header says how long until this request is affordable
                try:
                    time.sleep(max(1.0, float(r.headers.get("x-ratelimit-reset", 3))) + 0.2)
                except ValueError:
                    time.sleep(2)
                continue
            if r.status_code in (500, 502, 503, 504) and attempt < retries:
                time.sleep(1.5 * 2 ** attempt)
                continue
            if r.status_code >= 400:
                raise KalshiError(f"{method} {path} -> {r.status_code}: {r.text[:500]}")
            return r.json() if r.content else {}

    def paginate(self, path: str, key: str, params: dict | None = None, max_pages: int = 50):
        params = dict(params or {})
        for _ in range(max_pages):
            d = self.request("GET", path, params=params)
            yield from d.get(key) or []
            cur = d.get("cursor")
            if not cur:
                return
            params["cursor"] = cur

    # ---------------------------------------------------------------- market data (public)
    def markets(self, series_ticker: str, status: str = "open") -> list[dict]:
        return list(self.paginate("/markets", "markets", {
            "series_ticker": series_ticker, "status": status, "limit": 1000, "mve_filter": "exclude"}))

    def market(self, ticker: str) -> dict:
        return self.request("GET", f"/markets/{ticker}")["market"]

    def series(self, ticker: str) -> dict:
        return self.request("GET", f"/series/{ticker}")["series"]

    def collections(self, status: str = "open") -> list[dict]:
        return list(self.paginate("/multivariate_event_collections", "multivariate_contracts",
                                  {"status": status, "limit": 200}, max_pages=10))

    # ---------------------------------------------------------------- portfolio / combos (auth)
    def balance(self) -> float:
        d = self.request("GET", "/portfolio/balance", auth=True)
        return d.get("balance", 0) / 100.0

    def create_combo_market(self, collection_ticker: str, legs: list[dict]) -> dict:
        """legs: [{"market_ticker", "event_ticker", "side": "yes"}]. Idempotent for the same legs."""
        return self.request("POST", f"/multivariate_event_collections/{collection_ticker}",
                            json={"selected_markets": legs, "with_market_payload": True}, auth=True)

    def create_rfq(self, market_ticker: str, target_cost_dollars: float) -> str:
        d = self.request("POST", "/communications/rfqs", auth=True, json={
            "market_ticker": market_ticker,
            "target_cost_dollars": f"{target_cost_dollars:.2f}",
            "rest_remainder": False,
        })
        return d.get("id") or d.get("rfq_id") or d["rfq"]["id"]

    def quotes(self, rfq_id: str) -> list[dict]:
        d = self.request("GET", "/communications/quotes", params={"rfq_id": rfq_id}, auth=True)
        return d.get("quotes") or []

    def accept_quote(self, rfq_id: str, quote_id: str, side: str = "yes") -> dict:
        return self.request("PUT", f"/communications/rfqs/{rfq_id}/quotes/{quote_id}/accept",
                            json={"accepted_side": side}, auth=True)

    def delete_rfq(self, rfq_id: str) -> None:
        try:
            self.request("DELETE", f"/communications/rfqs/{rfq_id}", auth=True)
        except KalshiError:
            pass

    def fills(self, ticker: str | None = None) -> list[dict]:
        p = {"limit": 200}
        if ticker:
            p["ticker"] = ticker
        return self.request("GET", "/portfolio/fills", params=p, auth=True).get("fills") or []


def data_client() -> Kalshi:
    """Public prod market data (no key needed)."""
    return Kalshi(config.DATA_BASE_URL)


def trade_client() -> Kalshi:
    return Kalshi(config.TRADE_BASE_URL, config.KALSHI_KEY_ID, config.KALSHI_PRIVATE_KEY_PATH)


def fee(price: float, contracts: float = 1.0, mult: float = 1.0) -> float:
    """Kalshi taker fee: 0.07 * C * P * (1-P) (times the series' fee_multiplier)."""
    return 0.07 * mult * contracts * price * (1 - price)
