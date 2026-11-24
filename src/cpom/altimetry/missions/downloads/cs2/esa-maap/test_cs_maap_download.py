"""Tests for the download retry logic in cs_maap_download.py.

requests.get is replaced by a scripted fake server, so no network access or ESA login
is needed, and retry waits are recorded instead of slept.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path

import cs_maap_download as m  # pylint: disable=import-error
import pytest
import requests

URL = "https://catalog.maap.eo.esa.int/data/example/PRODUCT.nc"
BODY = b"x" * 1000


class FakeResponse:
    """Minimal stand-in for a streamed requests.Response."""

    def __init__(self, status: int, body: bytes = b"", headers: dict | None = None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}
        if status == 200:
            self.headers.setdefault("Content-Length", str(len(body)))
        self.text = body.decode(errors="replace")

    def iter_content(self, chunk_size):
        """Yield the body in chunks, like requests does."""
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeTokens:
    """TokenManager stand-in that counts logins."""

    def __init__(self):
        self.logins = 0
        self._token: str | None = None

    def get(self) -> str:
        """Return the current token, 'logging in' if there is none."""
        if self._token is None:
            self.logins += 1
            self._token = f"token{self.logins}"
        return self._token

    def invalidate(self) -> None:
        """Forget the token."""
        self._token = None


@pytest.fixture(name="server")
def fixture_server(monkeypatch):
    """Script the responses requests.get returns, in order; records the tokens sent."""
    script: list = []
    seen_tokens: list[str] = []

    def fake_get(url, headers=None, **_kw):
        assert url == URL
        seen_tokens.append(headers["Authorization"])
        nxt = script.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    monkeypatch.setattr(m.requests, "get", fake_get)
    return script, seen_tokens


def make_policy(retries=6, base_wait=30.0):
    """RetryPolicy that records waits instead of sleeping."""
    waits: list[float] = []
    return m.RetryPolicy(retries=retries, base_wait=base_wait, sleep=waits.append), waits


def test_intermittent_403_then_success(server, tmp_path: Path):
    """403s are retried with doubling waits, then the file downloads."""
    script, _ = server
    script += [FakeResponse(403, b"Forbidden"), FakeResponse(403), FakeResponse(200, BODY)]
    policy, waits = make_policy()
    dest = tmp_path / "PRODUCT.nc"
    m.download(URL, dest, FakeTokens(), len(BODY), policy)
    assert dest.read_bytes() == BODY
    assert waits == [30.0, 60.0]
    assert not (tmp_path / "PRODUCT.nc.part").exists()


def test_retry_after_seconds_is_honoured(server, tmp_path: Path):
    """A numeric Retry-After replaces the backoff wait."""
    script, _ = server
    script += [FakeResponse(429, headers={"Retry-After": "7"}), FakeResponse(200, BODY)]
    policy, waits = make_policy()
    m.download(URL, tmp_path / "PRODUCT.nc", FakeTokens(), len(BODY), policy)
    assert waits == [7.0]


def test_backoff_is_capped(server, tmp_path: Path):
    """The doubling wait stops at MAX_RETRY_WAIT."""
    script, _ = server
    script += [FakeResponse(503)] * 6 + [FakeResponse(200, BODY)]
    policy, waits = make_policy(retries=6, base_wait=100.0)
    m.download(URL, tmp_path / "PRODUCT.nc", FakeTokens(), len(BODY), policy)
    assert waits == [100.0, 200.0, 400.0, 600.0, 600.0, 600.0]


def test_gives_up_after_retries(server, tmp_path: Path):
    """After the allowed retries, the last error is raised with the server's reason."""
    script, _ = server
    script += [FakeResponse(403, b"quota exceeded")] * 3
    policy, waits = make_policy(retries=2)
    with pytest.raises(m.HTTPStatusError) as err:
        m.download(URL, tmp_path / "PRODUCT.nc", FakeTokens(), len(BODY), policy)
    assert err.value.status == 403
    assert "quota exceeded" in str(err.value)
    assert len(waits) == 2 and not script


def test_no_retries_when_disabled(server, tmp_path: Path):
    """--retries 0 fails on the first 403."""
    script, _ = server
    script += [FakeResponse(403)]
    policy, waits = make_policy(retries=0)
    with pytest.raises(m.HTTPStatusError):
        m.download(URL, tmp_path / "PRODUCT.nc", FakeTokens(), len(BODY), policy)
    assert not waits


def test_404_is_not_retried(server, tmp_path: Path):
    """A status outside RETRY_STATUSES fails at once."""
    script, _ = server
    script += [FakeResponse(404, b"Not Found")]
    policy, waits = make_policy()
    with pytest.raises(m.HTTPStatusError) as err:
        m.download(URL, tmp_path / "PRODUCT.nc", FakeTokens(), len(BODY), policy)
    assert err.value.status == 404 and not waits


def test_401_logs_in_again_once(server, tmp_path: Path):
    """A 401 gets one fresh login and an immediate retry, without waiting."""
    script, seen = server
    script += [FakeResponse(401), FakeResponse(200, BODY)]
    tokens = FakeTokens()
    policy, waits = make_policy()
    m.download(URL, tmp_path / "PRODUCT.nc", tokens, len(BODY), policy)
    assert tokens.logins == 2 and seen == ["Bearer token1", "Bearer token2"] and not waits


def test_repeated_401_fails(server, tmp_path: Path):
    """A second 401 after logging in again is a real authorisation failure."""
    script, _ = server
    script += [FakeResponse(401), FakeResponse(401)]
    policy, waits = make_policy()
    with pytest.raises(m.HTTPStatusError) as err:
        m.download(URL, tmp_path / "PRODUCT.nc", FakeTokens(), len(BODY), policy)
    assert err.value.status == 401 and not waits


def test_connection_error_is_retried(server, tmp_path: Path):
    """Dropped connections are retried like intermittent 403s."""
    script, _ = server
    script += [requests.ConnectionError("reset"), FakeResponse(200, BODY)]
    policy, waits = make_policy()
    m.download(URL, tmp_path / "PRODUCT.nc", FakeTokens(), len(BODY), policy)
    assert waits == [30.0]


def test_size_mismatch_still_detected(server, tmp_path: Path):
    """A short download is still an error, and leaves no final file."""
    script, _ = server
    script += [FakeResponse(200, BODY[:500])]
    policy, _ = make_policy()
    dest = tmp_path / "PRODUCT.nc"
    with pytest.raises(RuntimeError, match="Size mismatch"):
        m.download(URL, dest, FakeTokens(), len(BODY), policy)
    assert not dest.exists()


def test_parse_retry_after():
    """Retry-After may be seconds or an HTTP date; junk is ignored."""
    assert m.parse_retry_after("120") == 120.0
    assert m.parse_retry_after(None) is None
    assert m.parse_retry_after("soon") is None
    later = datetime.now(timezone.utc) + timedelta(seconds=90)
    assert 80 <= m.parse_retry_after(format_datetime(later, usegmt=True)) <= 91
    past = datetime.now(timezone.utc) - timedelta(seconds=90)
    assert m.parse_retry_after(format_datetime(past, usegmt=True)) == 0.0
