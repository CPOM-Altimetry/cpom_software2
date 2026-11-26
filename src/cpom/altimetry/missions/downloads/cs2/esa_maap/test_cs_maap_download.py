"""Tests for the download logic in cs_maap_download.py: retries, parallel downloads,
the per-file CSV log, the throughput summary and thread-safe login.

requests.get/post are replaced by scripted fakes, so no network access or ESA login
is needed, and retry waits are recorded instead of slept.
"""

from __future__ import annotations

import csv
import threading
import time
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path

import pytest
import requests

from cpom.altimetry.missions.downloads.cs2.esa_maap import cs_maap_download as m

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


# --- retries reported back -------------------------------------------------------------


def test_download_returns_retries_used(server, tmp_path: Path):
    """download() returns how many retries it needed."""
    script, _ = server
    script += [FakeResponse(503), FakeResponse(200, BODY)]
    policy, _ = make_policy()
    assert m.download(URL, tmp_path / "PRODUCT.nc", FakeTokens(), len(BODY), policy) == 1


def test_failure_carries_retries(server, tmp_path: Path):
    """A download that gives up records the retries used on the exception."""
    script, _ = server
    script += [FakeResponse(403)] * 3
    policy, _ = make_policy(retries=2)
    with pytest.raises(m.HTTPStatusError) as err:
        m.download(URL, tmp_path / "PRODUCT.nc", FakeTokens(), len(BODY), policy)
    assert getattr(err.value, "retries") == 2


# --- parallel runs, CSV log and summary ------------------------------------------------


@pytest.fixture(name="multi_server")
def fixture_multi_server(monkeypatch):
    """Fake server for many URLs: each URL has its own scripted responses."""
    script: dict[str, list] = {}
    lock = threading.Lock()
    active = {"now": 0, "max": 0}

    def fake_get(url, headers=None, **_kw):
        assert headers["Authorization"].startswith("Bearer ")
        with lock:
            nxt = script[url].pop(0)
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
        time.sleep(0.05)  # long enough for parallel requests to overlap
        with lock:
            active["now"] -= 1
        return nxt

    monkeypatch.setattr(m.requests, "get", fake_get)
    return script, active


def make_todo(script: dict, tmp_path: Path, n: int, failing: tuple[int, ...] = ()):
    """n products; those in `failing` get a 404, the rest download BODY."""
    todo = []
    for i in range(n):
        url = f"https://catalog.maap.eo.esa.int/data/example/P{i}.nc"
        script[url] = (
            [FakeResponse(404, b"Not Found")] if i in failing else [FakeResponse(200, BODY)]
        )
        todo.append((f"P{i}", url, tmp_path / "out" / f"P{i}.nc", len(BODY)))
    return todo


@pytest.mark.parametrize("workers", [1, 3])
def test_run_downloads_logs_every_file(multi_server, tmp_path: Path, workers: int):
    """Every file gets one CSV row, failures included, whatever the number of workers."""
    script, active = multi_server
    todo = make_todo(script, tmp_path, 6, failing=(2,))
    log_path = tmp_path / "log.csv"
    policy, _ = make_policy()
    results, wall = m.run_downloads(
        todo, FakeTokens(), policy, workers=workers, log=m.CsvLog(log_path)
    )
    assert wall > 0
    assert sorted(r.product_id for r in results) == [f"P{i}" for i in range(6)]
    assert [r.product_id for r in results if not r.ok] == ["P2"]
    assert all((tmp_path / "out" / f"P{i}.nc").read_bytes() == BODY for i in (0, 1, 3, 4, 5))
    rows = list(csv.DictReader(log_path.open()))
    assert len(rows) == 6 and list(rows[0]) == m.CSV_FIELDS
    failed = [r for r in rows if r["status"] == "failed"]
    assert len(failed) == 1 and failed[0]["product_id"] == "P2" and "404" in failed[0]["error"]
    assert all(r["start_utc"].endswith("Z") and float(r["seconds"]) > 0 for r in rows)
    assert all(int(r["bytes"]) == len(BODY) for r in rows if r["status"] == "ok")
    assert active["max"] == (1 if workers == 1 else 3)


def test_csv_log_appends_without_second_header(multi_server, tmp_path: Path):
    """A second run appends rows to an existing log, keeping one header."""
    script, _ = multi_server
    log = m.CsvLog(tmp_path / "log.csv")
    policy, _ = make_policy()
    m.run_downloads(make_todo(script, tmp_path / "a", 2), FakeTokens(), policy, log=log)
    m.run_downloads(make_todo(script, tmp_path / "b", 3), FakeTokens(), policy, log=log)
    lines = (tmp_path / "log.csv").read_text().splitlines()
    assert len(lines) == 1 + 5 and lines[0].startswith("start_utc,")


def test_login_error_stops_the_run(multi_server, tmp_path: Path):
    """A refused login is not a per-file failure: it stops the whole run."""
    script, _ = multi_server

    class RefusedTokens(FakeTokens):
        """Login always refused."""

        def get(self) -> str:
            raise m.LoginError("Login failed (401)")

    policy, _ = make_policy()
    with pytest.raises(m.LoginError):
        m.run_downloads(make_todo(script, tmp_path, 4), RefusedTokens(), policy, workers=2)


def test_summary_lines():
    """The summary reports counts, volume, rate and per-file times."""
    t0 = datetime(2026, 11, 25, tzinfo=timezone.utc)
    results = [
        m.FileResult(f"P{i}", Path("x"), t0, t0 + timedelta(seconds=i + 1), True, 2_000_000, 0)
        for i in range(4)
    ] + [m.FileResult("bad", Path("x"), t0, t0 + timedelta(seconds=1), False, 0, 6, "HTTP 403")]
    lines = m.summary_lines(results, wall=10.0, workers=2)
    assert lines[0] == "Done: 4 downloaded, 1 failed, 6 retries"
    assert "8.0 MB at 0.80 MB/s (24.0 files/min)" in lines[1] and "2 worker(s)" in lines[1]
    assert "median 2.50 s" in lines[2] and "max 4.00 s" in lines[2]


def test_percentile_and_duration():
    """Nearest-rank percentile and human-readable durations."""
    vals = [float(v) for v in range(1, 101)]
    assert m.percentile(vals, 95) == 95.0 and m.percentile(vals, 100) == 100.0
    assert m.percentile([3.0], 95) == 3.0
    assert m.fmt_duration(42) == "42.0 s"
    assert m.fmt_duration(725) == "12 min 05 s"
    assert m.fmt_duration(3 * 3600 + 7 * 60) == "3 h 07 min"


# --- thread-safe login ---------------------------------------------------------------


def test_token_manager_logs_in_once_for_parallel_callers(monkeypatch):
    """Threads asking for a token at the same time share one login."""
    calls = []

    class TokenResponse:  # pylint: disable=too-few-public-methods
        """Fake Keycloak token response."""

        status_code = 200
        text = ""

        @staticmethod
        def json():
            """Token JSON."""
            return {"access_token": "abc", "expires_in": 3600}

    def fake_post(*_a, **_kw):
        calls.append(1)
        time.sleep(0.05)
        return TokenResponse()

    monkeypatch.setattr(m.requests, "post", fake_post)
    tm = m.TokenManager({"grant_type": "password"}, "test")
    tokens: list[str] = []
    threads = [threading.Thread(target=lambda: tokens.append(tm.get())) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert tokens == ["abc"] * 8 and len(calls) == 1


def test_token_manager_raises_login_error(monkeypatch):
    """A refused login raises LoginError (it used to exit the program)."""

    class Refused:  # pylint: disable=too-few-public-methods
        """Fake 401 from Keycloak."""

        status_code = 401
        text = '{"error":"invalid_grant"}'

    monkeypatch.setattr(m.requests, "post", lambda *a, **k: Refused())
    with pytest.raises(m.LoginError, match="invalid_grant"):
        m.TokenManager({}, "test").get()
