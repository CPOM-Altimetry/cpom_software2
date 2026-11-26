#!/usr/bin/env python
r"""Search for and download CryoSat-2 products (L1b, L2, CryoTEMPO) from the ESA MAAP catalogue.

Search uses the public STAC API (no login needed). Downloads need a MAAP login, by
one of two methods. Either way the script gets short-lived access tokens from the
MAAP IAM endpoint and renews them as needed.

1. M2M service account (ESA MAAP DAS machine-to-machine ICD v1.2): a JSON file
       {"username": "...", "password": "...",
        "client_id": "...", "client_secret": "..."}
   located by: --m2m-file PATH, else $MAAP_M2M_FILE, else ~/.config/esa-maap/m2m.json
   (client_secret may be left out if ESA did not issue one)

2. Personal long-lasting (offline) token, generated at
       https://portal.maap.eo.esa.int/ini/services/auth/token/
   (valid 90 days) and saved as plain text, located by:
   --token-file PATH, else $MAAP_TOKEN_FILE, else ~/.config/esa-maap/token.txt

--auth auto (the default) uses the M2M file if it exists, otherwise the token file.
Keep both private, e.g.  chmod 600 ~/.config/esa-maap/*

Search-only runs (--dry-run) need no login.

Examples
--------
# List L1b SAR products over a bbox for one day (no download)
python cs_maap_download.py -c CryoSatIceL110 -p SIR_SAR_1B \
    --start 2025-01-01 --end 2025-01-02 --bbox -60 60 -20 85 -o BASE --dry-run

# Download the netCDF of up to 5 L2 SARIn products
python cs_maap_download.py -c CryoSatIceL210 -p SIR_SIN_2_ \
    --start 2025-01-01 --end 2025-01-02 -n 5 --asset nc -o BASE

# The single newest L1b SAR product
python cs_maap_download.py -c CryoSatIceL110 -p SIR_SAR_1B -n 1 -o BASE

# The last 3 days of L2 SARIn data available, into BASE/YYYY/MM/
python cs_maap_download.py -c CryoSatIceL210 -p SIR_SIN_2_ --last-days 3 -o BASE -m

# Regular update: anything added to the catalogue in the last 7 days that we don't have
python cs_maap_download.py -c CryoSatIceL210 -p SIR_SIN_2_ --new-days 7 -o BASE -m

Files already present (same name and size) are always skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Protocol

import requests
from pystac.utils import str_to_datetime
from pystac_client import Client
from pystac_client.stac_api_io import StacApiIO
from tqdm import tqdm
from urllib3.util.retry import Retry

CATALOG_URL = "https://catalog.maap.eo.esa.int/catalogue/"
# Download retries: on these HTTP statuses, or a dropped connection, wait and try again.
# 403 is included because MAAP downloads fail intermittently with HTTP 403 "You do not
# have the required credentials to access collection ..." even with a valid login: on
# mssldba (2026-11-24) this hit both the M2M service account and the offline token,
# about once in 6-20 files, and the same request succeeded on retry 30-120 s later.
# A real permission problem still fails, once the retries run out.
RETRY_STATUSES = {403, 429, 500, 502, 503, 504}
DEFAULT_RETRIES = 6
DEFAULT_RETRY_WAIT = 30.0  # seconds before the first retry; doubles each time
MAX_RETRY_WAIT = 600.0  # cap for the doubling (a server Retry-After is honoured as given)
PAGE_SIZE = 200  # items per catalogue request; 1000 timed out in testing, 500 worked
TOKEN_URL = "https://iam.maap.eo.esa.int/realms/esa-maap/protocol/openid-connect/token"
# Public client credentials published by ESA in the MAAP CryoSat data-access example
# (documented as permanently valid; they identify the offline-token client, not the user).
CLIENT_ID = "offline-token"
CLIENT_SECRET = "p1eL7uonXs6MDxtGbgKdPVRAmnGxHpVE"
TOKEN_FILE_ENV = "MAAP_TOKEN_FILE"
DEFAULT_TOKEN_FILE = Path.home() / ".config" / "esa-maap" / "token.txt"
M2M_FILE_ENV = "MAAP_M2M_FILE"
DEFAULT_M2M_FILE = Path.home() / ".config" / "esa-maap" / "m2m.json"
M2M_REQUIRED_KEYS = {"username", "password", "client_id"}
M2M_OPTIONAL_KEYS = {"client_secret"}

# Every CryoSat collection in the MAAP catalogue, with the complete product-type and
# baseline lists from each collection's /queryables endpoint (fetched 2026-11-21).
COLLECTIONS: dict[str, dict[str, Any]] = {
    "CryoSatIceL110": {
        "title": "CryoSat Ice Level 1",
        "types": {
            "SIR_LRM_1B": "L1b LRM",
            "SIR_SAR_1B": "L1b SAR",
            "SIR_SIN_1B": "L1b SARIn",
            "SIR1SAR_FR": "SAR full bit rate (FBR)",
            "SIR_SIN_FR": "SARIn full bit rate (FBR)",
        },
        "baselines": ["E001", "E002", "E003", "E004", "F001"],
    },
    "CryoSatIceL210": {
        "title": "CryoSat Ice L2",
        "types": {
            "SIR_LRM_2_": "L2 LRM",
            "SIR_SAR_2_": "L2 SAR",
            "SIR_SIN_2_": "L2 SARIn",
            "SIR_LRMI2_": "L2 intermediate (L2I) LRM",
            "SIR_SARI2_": "L2 intermediate (L2I) SAR",
            "SIR_SINI2_": "L2 intermediate (L2I) SARIn",
            "SIR_GDR_2_": "L2 GDR (all modes merged)",
        },
        "baselines": ["E001", "E002", "E003", "E004", "E005", "F001", "F002"],
    },
    "CryoSatOceanL110": {
        "title": "CryoSat Ocean L1",
        "types": {
            "SIR_GOPN1B": "L1b ocean, NOP/IOP/GOP timeliness variant",
            "SIR_GOPR1B": "L1b ocean, NOP/IOP/GOP timeliness variant",
            "SIR_GOPM1B": "L1b ocean, NOP/IOP/GOP timeliness variant",
        },
        "baselines": ["C001", "C002", "D001", "D002"],
    },
    "CryoSatOceanL210": {
        "title": "CryoSat Ocean L2",
        "types": {
            "SIR_GOPN_2": "L2 ocean, NOP/IOP/GOP timeliness variant",
            "SIR_GOPR_2": "L2 ocean, NOP/IOP/GOP timeliness variant",
            "SIR_GOPM_2": "L2 ocean, NOP/IOP/GOP timeliness variant",
            "SIR_GOP_2_": "L2 ocean, GOP",
        },
        "baselines": ["C001", "C002", "C003", "C004", "D001", "D002"],
    },
    "CryoTEMPO10": {
        "title": "CryoTEMPO",
        "types": {
            "SIR_TDP_LI": "Land ice",
            "SIR_TDP_SI": "Sea ice",
            "SIR_TDP_SS": "Summer sea ice",
            "SIR_TDP_PO": "Polar ocean",
            "SIR_TDP_CO": "Coastal ocean",
            "SIR_TDP_IW": "Inland water",
            "SIR_TDP_SD": "Monthly Arctic/Antarctic climatology grids",
        },
        "baselines": ["D001"],
    },
    "CryoTEMPOEOLIS10": {
        "title": "CryoTEMPO-EOLIS",
        "types": {
            "THEM_POINT_": "Swath elevation points",
            "THEM_GRID_": "Gridded elevation",
        },
        "baselines": ["V201", "V202", "V203", "V301", "V302", "V303"],
    },
}


def collections_help() -> str:
    """Help text listing every collection with its product types and baselines."""
    lines = ["CryoSat collections (-c), product types (-p) and baselines (-b):", ""]
    for cid, info in COLLECTIONS.items():
        lines.append(f"  {cid}  ({info['title']})")
        for t, desc in info["types"].items():
            lines.append(f"      {t:<12} {desc}")
        lines.append(f"      baselines: {', '.join(info['baselines'])}")
        lines.append("")
    lines.append("Lists taken from the catalogue's /queryables endpoint on 2026-11-21.")
    return "\n".join(lines)


ASSETS = {"nc": "enclosure_nc", "hdr": "enclosure_hdr", "zip": "product"}
ASSET_SUFFIX = {"nc": ".nc", "hdr": ".HDR", "zip": ".ZIP"}


def warn_if_not_private(path: Path) -> None:
    """Warn if a credentials file can be read by other users."""
    if path.stat().st_mode & 0o077:
        print(
            f"Warning: {path} can be read by other users; consider: chmod 600 {path}",
            file=sys.stderr,
        )


class LoginError(RuntimeError):
    """The MAAP login (token request) was refused."""


class TokenManager:
    """Get MAAP access tokens, reusing each until near expiry.

    Build one with from_token_file() (personal offline token) or from_m2m_file()
    (M2M service account). Only the login form differs between the two.
    """

    def __init__(self, form: dict[str, str], description: str):
        self._form = form
        self.description = description
        self._access: str | None = None
        self._expires = 0.0
        self._lock = threading.Lock()  # downloads may run in parallel (--workers)

    @classmethod
    def from_token_file(cls, token_file: Path) -> TokenManager:
        """Login with a personal long-lasting (offline) token, via the refresh_token grant."""
        if not token_file.is_file():
            sys.exit(
                f"Token file not found: {token_file}\n"
                "Generate a long-lasting token at "
                "https://portal.maap.eo.esa.int/ini/services/auth/token/ and save it there, "
                f"or point --token-file or ${TOKEN_FILE_ENV} at your token file."
            )
        warn_if_not_private(token_file)
        form = {
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "grant_type": "refresh_token",
            "refresh_token": token_file.read_text().strip(),
            "scope": "offline_access openid",
        }
        return cls(form, f"offline token ({token_file})")

    @classmethod
    def from_m2m_file(cls, m2m_file: Path) -> TokenManager:
        """Login with an M2M service account, via the password grant (ESA M2M ICD v1.2)."""
        if not m2m_file.is_file():
            sys.exit(
                f"M2M credentials file not found: {m2m_file}\n"
                "Create it as JSON with keys username, password, client_id "
                f"(and client_secret if issued), or point --m2m-file or ${M2M_FILE_ENV} at it."
            )
        warn_if_not_private(m2m_file)
        try:
            creds = json.loads(m2m_file.read_text())
        except json.JSONDecodeError as e:
            sys.exit(f"{m2m_file} is not valid JSON: {e.msg} (line {e.lineno})")
        if not isinstance(creds, dict):
            sys.exit(f"{m2m_file} must contain a JSON object")
        missing = M2M_REQUIRED_KEYS - creds.keys()
        unknown = creds.keys() - M2M_REQUIRED_KEYS - M2M_OPTIONAL_KEYS
        if missing or unknown:
            sys.exit(
                f"{m2m_file}: "
                + (f"missing keys {sorted(missing)} " if missing else "")
                + (f"unexpected keys {sorted(unknown)} " if unknown else "")
                + f"(expected {sorted(M2M_REQUIRED_KEYS)} plus optional client_secret)"
            )
        form = {
            "grant_type": "password",
            "client_id": str(creds["client_id"]),
            "username": str(creds["username"]),
            "password": str(creds["password"]),
            "scope": "openid",
        }
        if creds.get("client_secret"):
            form["client_secret"] = str(creds["client_secret"])
        return cls(form, f"M2M service account {creds['username']} ({m2m_file})")

    def invalidate(self) -> None:
        """Forget the current access token, so the next get() logs in again."""
        with self._lock:
            self._access = None

    def get(self) -> str:
        """Return a valid access token, requesting a new one if the last has (nearly) expired.

        Thread-safe: parallel downloads share one token, and only one thread logs in."""
        with self._lock:
            if self._access and time.time() < self._expires - 30:
                return self._access
            r = requests.post(TOKEN_URL, data=self._form, timeout=60)
            if r.status_code != 200:
                # Keycloak error bodies hold only an error code and description, no credentials.
                raise LoginError(
                    f"Login failed for {self.description} ({r.status_code}): {r.text[:300]}"
                )
            d = r.json()
            self._access = d["access_token"]
            self._expires = time.time() + float(d.get("expires_in", 60))
            return self._access


def resolve_path(cli_value: Path | None, env_var: str, default: Path) -> Path:
    """Path from the command line, else the environment variable, else the default."""
    if cli_value:
        return cli_value.expanduser()
    env = os.environ.get(env_var)
    if env:
        return Path(env).expanduser()
    return default


def make_token_manager(args) -> TokenManager:
    """Pick the login method from --auth (auto: M2M if its credentials file exists)."""
    m2m_file = resolve_path(args.m2m_file, M2M_FILE_ENV, DEFAULT_M2M_FILE)
    if args.auth == "m2m" or (args.auth == "auto" and m2m_file.is_file()):
        tokens = TokenManager.from_m2m_file(m2m_file)
    else:
        tokens = TokenManager.from_token_file(
            resolve_path(args.token_file, TOKEN_FILE_ENV, DEFAULT_TOKEN_FILE)
        )
    print(f"Login: {tokens.description}")
    return tokens


def build_filter(args, created_after: datetime | None = None) -> str | None:
    """CQL2-text filter from the product type, baseline, orbit and creation-date options."""
    clauses = []
    if args.product_type:
        clauses.append(f"\"product:type\"='{args.product_type}'")
    if args.baseline:
        clauses.append(f"version='{args.baseline.upper()}'")
    if args.orbit_state:
        clauses.append(f"\"sat:orbit_state\"='{args.orbit_state.upper()}'")
    if created_after:
        clauses.append(f"created>TIMESTAMP('{iso(created_after)}')")
    return " AND ".join(clauses) or None


def iso(t: datetime) -> str:
    """Format a datetime as a UTC ISO 8601 string, e.g. 2025-01-01T00:00:00Z."""
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def open_catalog():
    """STAC client that retries transient server errors (the catalogue sometimes returns 502)."""
    retry = Retry(
        total=6,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    return Client.open(CATALOG_URL, stac_io=StacApiIO(timeout=120, max_retries=retry))


def search(catalog, args, start=None, end=None, created_after=None, max_items=None):
    """Return an ItemSearch, newest sensing time first."""
    kw: dict[str, Any] = {
        "collections": [args.collection],
        "method": "GET",
        "max_items": max_items,
        "sortby": "-properties.datetime",
        "limit": PAGE_SIZE,
    }
    if start or end:
        kw["datetime"] = f"{start or '..'}/{end or '..'}"
    if args.bbox:
        kw["bbox"] = args.bbox
    flt = build_filter(args, created_after)
    if flt:
        kw["filter"] = flt
        kw["filter_lang"] = "cql2-text"
    return catalog.search(**kw)


def matches(item, args) -> bool:
    """Client-side check, in case the server ignores part of the filter."""
    if args.product_type and item.properties.get("product:type") != args.product_type:
        return False
    if args.baseline and str(item.properties.get("version", "")).upper() != args.baseline.upper():
        return False
    return True


def sensing_start(item):
    """Product sensing start time (UTC): start_datetime if given, else the item datetime."""
    start = item.properties.get("start_datetime")
    return str_to_datetime(start) if start else item.datetime


def dest_path(item, args) -> Path:
    """Local path for an item: OUTDIR[/YYYY/MM]/<product id><suffix>."""
    outdir = args.outdir
    if args.by_month:
        t = sensing_start(item)
        outdir = outdir / f"{t:%Y}" / f"{t:%m}"
    return outdir / (item.id + ASSET_SUFFIX[args.asset])


def have(dest: Path, expected_size: int | None) -> bool:
    """True if dest exists and (when the catalogue gives a size) has the expected size."""
    return dest.exists() and (expected_size is None or dest.stat().st_size == expected_size)


class TokenSource(Protocol):
    """What download() needs from a login: TokenManager, or a stand-in in tests."""

    def get(self) -> str:
        """Return a valid access token."""

    def invalidate(self) -> None:
        """Forget the current access token."""


class HTTPStatusError(RuntimeError):
    """A download request answered with an HTTP error status."""

    def __init__(self, status: int, url: str, reason: str, retry_after: str | None):
        self.status = status
        self.reason = reason
        self.retry_after = retry_after
        super().__init__(
            f"HTTP {status} for {url}\n      server says: {reason}"
            + (f"\n      Retry-After: {retry_after}" if retry_after else "")
        )


@dataclass
class RetryPolicy:
    """How download() retries: up to `retries` more attempts per file, waiting `base_wait`
    seconds before the first and doubling each time up to `max_wait`, or as long as the
    server's Retry-After header asks."""

    retries: int = DEFAULT_RETRIES
    base_wait: float = DEFAULT_RETRY_WAIT
    max_wait: float = MAX_RETRY_WAIT
    sleep: Callable[[float], None] = time.sleep

    def wait(self, attempt: int, retry_after: str | None) -> float:
        """Seconds to wait before retry number `attempt` (1 = first retry)."""
        server = parse_retry_after(retry_after)
        if server is not None:
            return server
        return min(self.base_wait * 2 ** (attempt - 1), self.max_wait)


def parse_retry_after(value: str | None) -> float | None:
    """Retry-After header as seconds from now; it may be a number or an HTTP date."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


def fetch(
    url: str, tmp: Path, tokens: TokenSource, expected_size: int | None, progress: bool = True
) -> None:
    """One attempt: stream url into tmp. Raises HTTPStatusError on an error status."""
    headers = {"Authorization": f"Bearer {tokens.get()}"}
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        if r.status_code >= 400:
            # Keep the server's reason and any Retry-After: they tell permission
            # problems apart from rate or quota limits.
            reason = " ".join(r.text[:300].split()) or "(empty body)"
            raise HTTPStatusError(r.status_code, url, reason, r.headers.get("Retry-After"))
        total = int(r.headers.get("Content-Length", 0)) or expected_size
        with (
            open(tmp, "wb") as f,
            tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=tmp.name[:50],
                leave=False,
                disable=not progress,
            ) as pbar,
        ):
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                pbar.update(len(chunk))


def with_retries(err: Exception, retries: int) -> Exception:
    """Attach how many retries were used to an exception, for the per-file log."""
    err.retries = retries  # type: ignore[attr-defined]
    return err


def download(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    url: str,
    dest: Path,
    tokens: TokenSource,
    expected_size: int | None,
    policy: RetryPolicy | None = None,
    progress: bool = True,
) -> int:
    """Download url to dest via a .part file, retrying intermittent and transient errors,
    then check the size against the catalogue.

    Returns the number of retries used. On failure, the exception raised carries the
    same count in its `retries` attribute."""
    policy = policy or RetryPolicy()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    relogged = False
    attempt = 0
    while True:
        what, retry_after = "", None
        try:
            fetch(url, tmp, tokens, expected_size, progress)
            break
        except HTTPStatusError as e:
            if e.status == 401 and not relogged:
                # The access token may have been revoked or expired early: log in again once.
                relogged = True
                tokens.invalidate()
                continue
            if e.status not in RETRY_STATUSES or attempt >= policy.retries:
                with_retries(e, attempt)
                raise
            what, retry_after = f"HTTP {e.status} ({e.reason[:120]})", e.retry_after
        except (
            requests.ConnectionError,
            requests.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ) as e:
            if attempt >= policy.retries:
                raise with_retries(
                    RuntimeError(f"{type(e).__name__} for {url}: {e}"), attempt
                ) from e
            what, retry_after = type(e).__name__, None
        attempt += 1
        wait = policy.wait(attempt, retry_after)
        print(
            f"  {dest.name}: {what}; retry {attempt}/{policy.retries} in {wait:.0f} s",
            file=sys.stderr,
        )
        policy.sleep(wait)
    if expected_size and tmp.stat().st_size != expected_size:
        raise with_retries(
            RuntimeError(
                f"Size mismatch for {dest.name}: got {tmp.stat().st_size}, "
                f"expected {expected_size}"
            ),
            attempt,
        )
    tmp.rename(dest)
    return attempt


CSV_FIELDS = [
    "start_utc",
    "end_utc",
    "product_id",
    "status",
    "bytes",
    "seconds",
    "mb_per_s",
    "retries",
    "error",
]


@dataclass
class FileResult:  # pylint: disable=too-many-instance-attributes
    """Outcome of one file download, as logged to the CSV file."""

    product_id: str
    dest: Path
    start: datetime
    end: datetime
    ok: bool
    nbytes: int
    retries: int
    error: str = ""

    @property
    def seconds(self) -> float:
        """Wall-clock time for this file, including any retry waits."""
        return (self.end - self.start).total_seconds()

    def csv_row(self) -> dict[str, Any]:
        """This result as a row for the --log-csv file."""
        return {
            "start_utc": iso_ms(self.start),
            "end_utc": iso_ms(self.end),
            "product_id": self.product_id,
            "status": "ok" if self.ok else "failed",
            "bytes": self.nbytes,
            "seconds": f"{self.seconds:.3f}",
            "mb_per_s": f"{self.nbytes / 1e6 / self.seconds:.3f}" if self.seconds > 0 else "",
            "retries": self.retries,
            "error": self.error,
        }


def iso_ms(t: datetime) -> str:
    """UTC ISO 8601 with milliseconds, e.g. 2026-11-25T10:00:00.123Z."""
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


@dataclass
class CsvLog:
    """Thread-safe append-only per-file CSV log (header written if the file is new)."""

    path: Path | None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def write(self, result: FileResult) -> None:
        """Append one row, flushed straight away so a crash loses nothing."""
        if self.path is None:
            return
        with self._lock:
            new = not self.path.exists() or self.path.stat().st_size == 0
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                if new:
                    w.writeheader()
                w.writerow(result.csv_row())


def download_one(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    product_id: str,
    url: str,
    dest: Path,
    size: int | None,
    tokens: TokenSource,
    policy: RetryPolicy,
    progress: bool,
) -> FileResult:
    """Download one file and time it; never raises for a failed download."""
    start = datetime.now(timezone.utc)
    try:
        retries = download(url, dest, tokens, size, policy, progress)
        return FileResult(
            product_id, dest, start, datetime.now(timezone.utc), True, dest.stat().st_size, retries
        )
    except (requests.RequestException, RuntimeError, OSError) as e:
        if isinstance(e, LoginError):
            raise  # a refused login fails every file: stop the run
        error = " ".join(str(e).split())
        return FileResult(
            product_id,
            dest,
            start,
            datetime.now(timezone.utc),
            False,
            0,
            getattr(e, "retries", 0),
            error,
        )


def run_downloads(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    todo: list[tuple[str, str, Path, int | None]],
    tokens: TokenSource,
    policy: RetryPolicy,
    workers: int = 1,
    log: CsvLog | None = None,
    report: Callable[[FileResult], None] | None = None,
) -> tuple[list[FileResult], float]:
    """Download (product_id, url, dest, size) entries, `workers` at a time.

    Returns the per-file results (in completion order) and the total wall-clock seconds."""
    log = log or CsvLog(None)
    results: list[FileResult] = []
    progress = workers == 1  # per-file progress bars only make sense one at a time
    t0 = time.monotonic()

    def done(res: FileResult) -> None:
        results.append(res)
        log.write(res)
        if report:
            report(res)

    if workers == 1:
        for pid, url, dest, size in todo:
            done(download_one(pid, url, dest, size, tokens, policy, progress))
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(download_one, pid, url, dest, size, tokens, policy, progress)
                for pid, url, dest, size in todo
            ]
            try:
                for fut in as_completed(futures):
                    done(fut.result())
            except BaseException:
                ex.shutdown(wait=False, cancel_futures=True)  # e.g. Ctrl-C: don't start more
                raise
    return results, time.monotonic() - t0


def summary_lines(results: list[FileResult], wall: float, workers: int) -> list[str]:
    """Throughput summary for the end of a run."""
    ok = [r for r in results if r.ok]
    nbytes = sum(r.nbytes for r in ok)
    lines = [
        f"Done: {len(ok)} downloaded, {len(results) - len(ok)} failed, "
        f"{sum(r.retries for r in results)} retries",
        (
            f"Time: {fmt_duration(wall)} wall clock with {workers} worker(s); "
            f"{nbytes / 1e6:.1f} MB at {nbytes / 1e6 / wall:.2f} MB/s"
            f" ({len(ok) / wall * 60:.1f} files/min)"
            if wall > 0 and ok
            else ""
        ),
    ]
    if ok:
        secs = sorted(r.seconds for r in ok)
        rates = sorted(r.nbytes / 1e6 / r.seconds for r in ok if r.seconds > 0)
        lines.append(
            f"Per file: median {statistics.median(secs):.2f} s, "
            f"95th percentile {percentile(secs, 95):.2f} s, max {secs[-1]:.2f} s"
            + (f"; median {statistics.median(rates):.2f} MB/s" if rates else "")
        )
    return [line for line in lines if line]


def percentile(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile of an already sorted, non-empty list."""
    k = max(0, min(len(sorted_values) - 1, math.ceil(pct / 100 * len(sorted_values)) - 1))
    return sorted_values[k]


def fmt_duration(seconds: float) -> str:
    """Seconds as e.g. '42.0 s', '12 min 05 s' or '3 h 07 min'."""
    if seconds < 60:
        return f"{seconds:.1f} s"
    m, s = divmod(round(seconds), 60)
    if m < 60:
        return f"{m} min {s:02d} s"
    h, m = divmod(m, 60)
    return f"{h} h {m:02d} min"


def main(argv=None) -> int:
    """Command-line entry point. Returns 0 on success, 1 if any download failed."""
    p = argparse.ArgumentParser(
        description=__doc__,
        epilog=collections_help(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "-c",
        "--collection",
        required=True,
        choices=list(COLLECTIONS),
        metavar="COLLECTION",
        help="collection ID (full list below)",
    )
    p.add_argument(
        "-p", "--product-type", help="product type within the collection (full list below)"
    )
    p.add_argument("-b", "--baseline", help="processing baseline, e.g. E001 (full list below)")
    p.add_argument("--orbit-state", choices=["ascending", "descending"])
    p.add_argument("--bbox", nargs=4, type=float, metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"))
    g = p.add_argument_group("time selection (choose one way; default: no time limit)")
    g.add_argument(
        "--start", help="sensing start date/time, e.g. 2025-01-01 or 2025-01-01T06:00:00Z"
    )
    g.add_argument("--end", help="sensing end date/time (a date alone covers that whole day)")
    g.add_argument(
        "-l",
        "--last-days",
        type=float,
        metavar="N",
        help="the most recent N days of data AVAILABLE in the catalogue, counted back from "
        "the newest product matching the other filters (not from today, as the "
        "catalogue lags real time by weeks)",
    )
    g.add_argument(
        "--new-days",
        type=float,
        metavar="N",
        help="products ADDED to the catalogue in the last N days (by catalogue creation "
        "date), whatever their sensing date; useful for regular update runs",
    )
    p.add_argument(
        "-n",
        "--max-items",
        type=int,
        help="max products to search (newest first); 0 = no limit. "
        "Default: 10, or no limit with --last-days/--new-days",
    )
    p.add_argument(
        "-a",
        "--asset",
        choices=list(ASSETS),
        default="nc",
        help="nc = netCDF, hdr = header, zip = full product ZIP (default: nc)",
    )
    p.add_argument(
        "-o", "--outdir", type=Path, required=True, help="output (base) directory (required)"
    )
    p.add_argument(
        "-m",
        "--by-month",
        action="store_true",
        help="save each file in OUTDIR/YYYY/MM/, using the product's sensing start time; "
        "directories are created as needed",
    )
    login = p.add_argument_group("login (downloads only; see the description above)")
    login.add_argument(
        "--auth",
        choices=["auto", "m2m", "token"],
        default="auto",
        help="login method; auto (default) uses the M2M file if it exists, else the token file",
    )
    login.add_argument(
        "--m2m-file",
        type=Path,
        help=f"M2M service-account JSON file (default: ${M2M_FILE_ENV}, else {DEFAULT_M2M_FILE})",
    )
    login.add_argument(
        "-t",
        "--token-file",
        type=Path,
        help=f"offline token file (default: ${TOKEN_FILE_ENV}, else {DEFAULT_TOKEN_FILE})",
    )
    p.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help="per file, how many times to retry after HTTP 403/429/5xx or a dropped "
        f"connection (default: {DEFAULT_RETRIES}; 0 = no retries)",
    )
    p.add_argument(
        "--retry-wait",
        type=float,
        default=DEFAULT_RETRY_WAIT,
        metavar="SECONDS",
        help=f"wait before the first retry, doubling each time up to {MAX_RETRY_WAIT:.0f} s; "
        f"a server Retry-After is used instead when given (default: {DEFAULT_RETRY_WAIT:.0f})",
    )
    p.add_argument(
        "-w",
        "--workers",
        type=int,
        default=1,
        help="download this many files at once (default: 1). Agree any value above 1 with "
        "ESA first",
    )
    p.add_argument(
        "--log-csv",
        type=Path,
        metavar="FILE",
        help="append one line per file to this CSV: UTC start/end, product, status, bytes, "
        "seconds, MB/s, retries, error",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="search and list what would be downloaded; no login needed",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true", help="also list products already downloaded"
    )
    args = p.parse_args(argv)

    if args.last_days is not None and (args.start or args.end):
        p.error("--last-days cannot be combined with --start/--end")
    if args.workers < 1:
        p.error("--workers must be at least 1")
    info = COLLECTIONS[args.collection]
    # Warn rather than fail: ESA may add new types or baselines after this list was made.
    if args.product_type and args.product_type not in info["types"]:
        print(
            f"Warning: {args.product_type} is not a known type for {args.collection}: "
            f"{', '.join(info['types'])}",
            file=sys.stderr,
        )
    if args.baseline and args.baseline.upper() not in info["baselines"]:
        print(
            f"Warning: {args.baseline} is not a known baseline for {args.collection}: "
            f"{', '.join(info['baselines'])}",
            file=sys.stderr,
        )

    windowed = args.last_days is not None or args.new_days is not None
    max_items = args.max_items if args.max_items is not None else (0 if windowed else 10)
    max_items = max_items or None  # 0 -> no limit

    catalog = open_catalog()
    start, end = args.start, args.end
    created_after = None
    if args.new_days is not None:
        created_after = datetime.now(timezone.utc) - timedelta(days=args.new_days)
        print(f"Products added to the catalogue since {iso(created_after)}")
    if args.last_days is not None:
        newest = next(search(catalog, args, created_after=created_after, max_items=1).items(), None)
        if newest is None:
            print("No products match these filters.")
            return 0
        start = iso(newest.datetime - timedelta(days=args.last_days))
        lag = (datetime.now(timezone.utc) - newest.datetime).days
        print(f"Newest product: {newest.id} (sensed {iso(newest.datetime)}, {lag} days ago)")
        print(f"Sensing window: {start} to newest")

    s = search(catalog, args, start, end, created_after, max_items)
    try:
        total = s.matched()
    except Exception:  # pylint: disable=broad-exception-caught
        total = None  # the total is only informative; carry on without it
    asset_key = ASSETS[args.asset]
    todo, n_have, n_noasset, n_seen = [], 0, 0, 0
    for it in s.items():
        if not matches(it, args):
            continue
        n_seen += 1
        a = it.assets.get(asset_key)
        if a is None:
            n_noasset += 1
            continue
        size = a.extra_fields.get("file:size")
        dest = dest_path(it, args)
        line = (
            f"  {it.id}  {it.properties.get('product:type')}  {it.datetime:%Y-%m-%dT%H:%M:%S}"
            + (f"  {size/1e6:.1f} MB" if size else "")
        )
        if have(dest, size):
            n_have += 1
            if args.verbose:
                print(line + "  [have]")
            continue
        todo.append((it, a, dest, size))
        print(line)

    known = [x[3] for x in todo if x[3]]
    vol = f", {sum(known)/1e6:.1f} MB" + ("" if len(known) == len(todo) else "+") if known else ""
    print(
        f"{args.collection}: {n_seen} product(s) found"
        + (
            f" (of {total} matching; raise -n to see more)"
            if total and max_items and total > n_seen
            else ""
        )
        + f"; {n_have} already downloaded; {len(todo)} to download{vol}"
        + (f"; {n_noasset} without a '{asset_key}' asset" if n_noasset else "")
    )
    if args.dry_run or not todo:
        return 0

    tokens = make_token_manager(args)
    policy = RetryPolicy(retries=args.retries, base_wait=args.retry_wait)

    def report(res: FileResult) -> None:
        if res.ok:
            print(f"  downloaded: {res.dest}  ({res.nbytes / 1e6:.1f} MB, {res.seconds:.1f} s)")
        else:
            print(f"  FAILED {res.product_id}: {res.error}", file=sys.stderr)

    try:
        results, wall = run_downloads(
            [(it.id, a.href, dest, size) for it, a, dest, size in todo],
            tokens,
            policy,
            workers=args.workers,
            log=CsvLog(args.log_csv),
            report=report,
        )
    except LoginError as e:
        sys.exit(str(e))
    for line in summary_lines(results, wall, args.workers):
        print(line)
    if args.log_csv:
        print(f"Per-file log: {args.log_csv}")
    return 1 if any(not r.ok for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
