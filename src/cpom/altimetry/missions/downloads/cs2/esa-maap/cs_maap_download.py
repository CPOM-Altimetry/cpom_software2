#!/usr/bin/env python
r"""Search for and download CryoSat-2 products (L1b, L2, CryoTEMPO) from the ESA MAAP catalogue.

Search uses the public STAC API (no login needed). Downloads need an ESA MAAP
long-lasting (offline) token, generated at
    https://portal.maap.eo.esa.int/ini/services/auth/token/
and saved as plain text in a token file. The script exchanges it for a short-lived
access token at the MAAP IAM endpoint and refreshes it as needed.

Token file location, first match wins:
    1. --token-file PATH
    2. the MAAP_TOKEN_FILE environment variable
    3. ~/.config/esa-maap/token.txt
Keep it private:  chmod 600 ~/.config/esa-maap/token.txt

Search-only runs (--dry-run) need no token.

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
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
from pystac.utils import str_to_datetime
from pystac_client import Client
from pystac_client.stac_api_io import StacApiIO
from tqdm import tqdm
from urllib3.util.retry import Retry

CATALOG_URL = "https://catalog.maap.eo.esa.int/catalogue/"
PAGE_SIZE = 200  # items per catalogue request; 1000 timed out in testing, 500 worked
TOKEN_URL = "https://iam.maap.eo.esa.int/realms/esa-maap/protocol/openid-connect/token"
# Public client credentials published by ESA in the MAAP CryoSat data-access example
# (documented as permanently valid; they identify the offline-token client, not the user).
CLIENT_ID = "offline-token"
CLIENT_SECRET = "p1eL7uonXs6MDxtGbgKdPVRAmnGxHpVE"
TOKEN_FILE_ENV = "MAAP_TOKEN_FILE"
DEFAULT_TOKEN_FILE = Path.home() / ".config" / "esa-maap" / "token.txt"

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


class TokenManager:  # pylint: disable=too-few-public-methods
    """Exchange the offline token for access tokens, reusing each until near expiry."""

    def __init__(self, token_file: Path):
        if not token_file.is_file():
            sys.exit(
                f"Token file not found: {token_file}\n"
                "Generate a long-lasting token at "
                "https://portal.maap.eo.esa.int/ini/services/auth/token/ and save it there, "
                f"or point --token-file or ${TOKEN_FILE_ENV} at your token file."
            )
        if token_file.stat().st_mode & 0o077:
            print(
                f"Warning: {token_file} can be read by other users; "
                f"consider: chmod 600 {token_file}",
                file=sys.stderr,
            )
        self.offline_token = token_file.read_text().strip()
        self._access: str | None = None
        self._expires = 0.0

    def get(self) -> str:
        """Return a valid access token, requesting a new one if the last has (nearly) expired."""
        if self._access and time.time() < self._expires - 30:
            return self._access
        r = requests.post(
            TOKEN_URL,
            data={
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "grant_type": "refresh_token",
                "refresh_token": self.offline_token,
                "scope": "offline_access openid",
            },
            timeout=60,
        )
        if r.status_code != 200:
            sys.exit(f"Token exchange failed ({r.status_code}): {r.text[:300]}")
        d = r.json()
        self._access = d["access_token"]
        self._expires = time.time() + float(d.get("expires_in", 60))
        return self._access


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


def resolve_token_file(cli_value: Path | None) -> Path:
    """Token file from --token-file, else $MAAP_TOKEN_FILE, else ~/.config/esa-maap/token.txt."""
    if cli_value:
        return cli_value.expanduser()
    env = os.environ.get(TOKEN_FILE_ENV)
    if env:
        return Path(env).expanduser()
    return DEFAULT_TOKEN_FILE


def download(url: str, dest: Path, tokens: TokenManager, expected_size: int | None) -> None:
    """Stream url to dest via a .part file, checking the size against the catalogue."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    headers = {"Authorization": f"Bearer {tokens.get()}"}
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        if r.status_code in (401, 403):
            raise RuntimeError(f"HTTP {r.status_code} (not authorised) for {url}")
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) or expected_size
        with (
            open(tmp, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc=dest.name[:50], leave=False) as pbar,
        ):
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                pbar.update(len(chunk))
    if expected_size and tmp.stat().st_size != expected_size:
        raise RuntimeError(
            f"Size mismatch for {dest.name}: got {tmp.stat().st_size}, expected {expected_size}"
        )
    tmp.rename(dest)


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
    p.add_argument(
        "-t",
        "--token-file",
        type=Path,
        help=f"MAAP offline token file (default: ${TOKEN_FILE_ENV}, " f"else {DEFAULT_TOKEN_FILE})",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="search and list what would be downloaded; no token needed",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true", help="also list products already downloaded"
    )
    args = p.parse_args(argv)

    if args.last_days is not None and (args.start or args.end):
        p.error("--last-days cannot be combined with --start/--end")
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

    tokens = TokenManager(resolve_token_file(args.token_file))
    failures = 0
    for it, a, dest, size in todo:
        try:
            download(a.href, dest, tokens, size)
            print(f"  downloaded: {dest}")
        except (requests.RequestException, RuntimeError, OSError) as e:  # keep going
            failures += 1
            print(f"  FAILED {it.id}: {e}", file=sys.stderr)
    print(f"Done: {len(todo) - failures} downloaded, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
