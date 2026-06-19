from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


@dataclass
class HttpResponse:
    content: bytes
    encoding: str = "utf-8"

    @property
    def text(self) -> str:
        return self.content.decode(self.encoding or "utf-8", errors="replace")

    def json(self):
        return json.loads(self.text)


try:
    import requests
except ModuleNotFoundError:
    requests = None


def get(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, str] | None = None,
    timeout: int = 60,
) -> HttpResponse:
    headers = headers or {}

    if requests is not None:
        response = requests.get(url, headers=headers, params=params, timeout=timeout)
        response.raise_for_status()
        encoding = response.encoding or "utf-8"
        return HttpResponse(content=response.content, encoding=encoding)

    if params:
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}{urlencode(params)}"

    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get_content_charset() or "utf-8"
            return HttpResponse(content=response.read(), encoding=content_type)
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} while fetching {url}") from exc
