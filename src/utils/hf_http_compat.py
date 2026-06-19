from __future__ import annotations

import io
import os
import ssl
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any


_PATCHED = False
_CLIENT_CONFIGURED = False


def _env_enabled(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "on"}


def _ca_bundle_path() -> str | None:
    for name in ("HF_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE"):
        value = os.environ.get(name)
        if value:
            return value
    return None


def _build_verify_config():
    ca_bundle = _ca_bundle_path()
    if ca_bundle:
        path = Path(ca_bundle).expanduser()
        if path.exists():
            return str(path)

    if _env_enabled("HF_HUB_DISABLE_SSL_VERIFY"):
        return False

    try:
        import truststore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except Exception:
        return True


def configure_huggingface_http_client() -> bool:
    """Make Hugging Face Hub's httpx client honor corporate/system CAs."""
    global _CLIENT_CONFIGURED

    if _CLIENT_CONFIGURED or os.environ.get("HF_HTTP_CLIENT_COMPAT", "1").lower() in {"0", "false", "no", "off"}:
        return _CLIENT_CONFIGURED

    try:
        import httpx
        from huggingface_hub.utils import _http as hf_http
    except Exception:
        return False

    verify = _build_verify_config()

    def client_factory() -> httpx.Client:
        return httpx.Client(
            event_hooks={"request": [hf_http.hf_request_event_hook]},
            follow_redirects=True,
            timeout=None,
            verify=verify,
        )

    def async_client_factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(
            event_hooks={
                "request": [hf_http.async_hf_request_event_hook],
                "response": [hf_http.async_hf_response_event_hook],
            },
            follow_redirects=True,
            timeout=None,
            verify=verify,
        )

    hf_http.set_client_factory(client_factory)
    hf_http.set_async_client_factory(async_client_factory)
    _CLIENT_CONFIGURED = True
    return True


def _is_closed_client_error(err: RuntimeError) -> bool:
    return "client has been closed" in str(err).lower()


def patch_huggingface_http_backoff() -> bool:
    """Work around a Hugging Face Hub retry path that can reuse a closed httpx client."""
    global _PATCHED

    if _PATCHED or os.environ.get("HF_HTTP_COMPAT_PATCH", "1").lower() in {"0", "false", "no", "off"}:
        return _PATCHED

    try:
        import httpx
        from huggingface_hub.utils import _http as hf_http
        from huggingface_hub.utils._lfs import SliceFileObj
    except Exception:
        return False

    original = getattr(hf_http, "_http_backoff_base", None)
    if original is None or getattr(original, "_llama32_closed_client_patch", False):
        _PATCHED = True
        return True

    def patched_http_backoff_base(
        method,
        url: str,
        *,
        max_retries: int = 5,
        base_wait_time: float = 1,
        max_wait_time: float = 8,
        retry_on_exceptions=hf_http._DEFAULT_RETRY_ON_EXCEPTIONS,
        retry_on_status_codes=hf_http._DEFAULT_RETRY_ON_STATUS_CODES,
        stream: bool = False,
        **kwargs: Any,
    ) -> Generator[httpx.Response, None, None]:
        if isinstance(retry_on_exceptions, type):
            retry_on_exceptions = (retry_on_exceptions,)
        if isinstance(retry_on_status_codes, int):
            retry_on_status_codes = (retry_on_status_codes,)

        nb_tries = 0
        sleep_time = base_wait_time
        ratelimit_reset: int | None = None
        io_obj_initial_pos = None

        if "data" in kwargs and isinstance(kwargs["data"], (io.IOBase, SliceFileObj)):
            io_obj_initial_pos = kwargs["data"].tell()

        while True:
            nb_tries += 1
            ratelimit_reset = None
            client = hf_http.get_session()
            try:
                if io_obj_initial_pos is not None:
                    kwargs["data"].seek(io_obj_initial_pos)

                def should_retry(response: httpx.Response) -> bool:
                    nonlocal ratelimit_reset

                    if response.status_code not in retry_on_status_codes:
                        return False

                    hf_http.logger.warning(f"HTTP Error {response.status_code} thrown while requesting {method} {url}")
                    if nb_tries > max_retries:
                        hf_http.hf_raise_for_status(response)
                        return False

                    if response.status_code == 429:
                        ratelimit_info = hf_http.parse_ratelimit_headers(response.headers)
                        if ratelimit_info is not None:
                            ratelimit_reset = ratelimit_info.reset_in_seconds

                    return True

                if stream:
                    with client.stream(method=method, url=url, **kwargs) as response:
                        if not should_retry(response):
                            yield response
                            return
                else:
                    response = client.request(method=method, url=url, **kwargs)
                    if not should_retry(response):
                        yield response
                        return

            except RuntimeError as err:
                if not _is_closed_client_error(err):
                    raise
                hf_http.logger.warning(f"Closed Hugging Face HTTP client while requesting {method} {url}; recreating it.")
                hf_http.close_session()
                if nb_tries > max_retries:
                    raise

            except retry_on_exceptions as err:
                hf_http.logger.warning(f"'{err}' thrown while requesting {method} {url}")

                if isinstance(err, httpx.ConnectError):
                    hf_http.close_session()

                if nb_tries > max_retries:
                    raise err

            if ratelimit_reset is not None:
                actual_sleep = float(ratelimit_reset) + 1
                hf_http.logger.warning(f"Rate limited. Waiting {actual_sleep}s before retry [Retry {nb_tries}/{max_retries}].")
            else:
                actual_sleep = sleep_time
                hf_http.logger.warning(f"Retrying in {actual_sleep}s [Retry {nb_tries}/{max_retries}].")

            time.sleep(actual_sleep)
            sleep_time = min(max_wait_time, sleep_time * 2)

    patched_http_backoff_base._llama32_closed_client_patch = True
    hf_http._http_backoff_base = patched_http_backoff_base
    _PATCHED = True
    return True
