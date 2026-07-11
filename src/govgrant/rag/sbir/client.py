"""HTTP client for SBIR.gov Solicitation / Topic API."""

from __future__ import annotations

from typing import Any

import httpx

from govgrant.rag.config import Settings, get_settings


class SBIRAPIError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class SBIRTopicClient:
    """
    Client for https://api.www.sbir.gov/public/api/solicitations

    Auth: optional API key via:
      - header X-API-Key
      - header Authorization: Bearer <key>
      - query param api_key (legacy)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.base_url = self.settings.sbir_api_base_url.rstrip("/")
        self.api_key = self.settings.sbir_api_key
        self.timeout = self.settings.sbir_timeout_seconds

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "User-Agent": "GovGrant-AI/0.1 (sbir-connector)",
        }
        if self.api_key:
            headers["X-API-Key"] = self.api_key
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _params(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"format": "json"}
        if extra:
            params.update({k: v for k, v in extra.items() if v is not None})
        if self.api_key:
            params.setdefault("api_key", self.api_key)
        return params

    def fetch_solicitations(
        self,
        *,
        open_only: bool = True,
        keyword: str | None = None,
        agency: str | None = None,
        rows: int = 100,
        start: int = 0,
    ) -> list[dict[str, Any]]:
        """Fetch one page of solicitations."""
        params = self._params(
            {
                "open": 1 if open_only else None,
                "keyword": keyword,
                "agency": agency,
                "rows": rows,
                "start": start,
            }
        )
        url = f"{self.base_url}/solicitations"
        try:
            with httpx.Client(timeout=self.timeout, headers=self._headers()) as client:
                resp = client.get(url, params=params)
        except httpx.HTTPError as exc:
            raise SBIRAPIError(f"SBIR network error: {exc}") from exc

        if resp.status_code == 403:
            raise SBIRAPIError(
                "SBIR API returned 403 Forbidden (maintenance or missing/invalid API key). "
                "Set SBIR_API_KEY or use fixtures.",
                status_code=403,
            )
        if resp.status_code >= 400:
            raise SBIRAPIError(
                f"SBIR API error {resp.status_code}: {resp.text[:300]}",
                status_code=resp.status_code,
            )

        data = resp.json()
        if isinstance(data, dict):
            if data.get("message") in {"Forbidden", "Unauthorized"}:
                raise SBIRAPIError(
                    f"SBIR API message: {data.get('message')}",
                    status_code=resp.status_code,
                )
            # some wrappers
            for key in ("results", "data", "solicitations", "items"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            raise SBIRAPIError(f"Unexpected SBIR response shape: {list(data.keys())}")
        if isinstance(data, list):
            return data
        raise SBIRAPIError(f"Unexpected SBIR response type: {type(data)}")

    def fetch_all_open(
        self,
        *,
        keyword: str | None = None,
        agency: str | None = None,
        page_size: int = 100,
        max_pages: int = 50,
    ) -> list[dict[str, Any]]:
        """Paginate open solicitations."""
        all_rows: list[dict[str, Any]] = []
        start = 0
        for _ in range(max_pages):
            page = self.fetch_solicitations(
                open_only=True,
                keyword=keyword,
                agency=agency,
                rows=page_size,
                start=start,
            )
            if not page:
                break
            all_rows.extend(page)
            if len(page) < page_size:
                break
            start += page_size
        return all_rows
