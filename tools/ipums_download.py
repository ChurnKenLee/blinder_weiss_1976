#!/usr/bin/env python3
"""Retrieve IPUMS metadata and authorized USA/ATUS extracts, without logging keys.

Run ``python tools/ipums_download.py --help``. API credentials are read from
IPUMS_API_KEY, IPUMS_API_KEY_FILE, or the private file /tmp/ipums_api_key.
Only the explicit ``submit`` command creates a new extract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/ipums"
API_HOST = "api.ipums.org"


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.parent / ".staging"
    staging.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=staging, mode="w", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def public_url(url):
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.hostname or "", parts.path, "", ""))


def is_ipums_host(host):
    return bool(host and (host == "ipums.org" or host.endswith(".ipums.org")
                         or host == "atusdata.org" or host.endswith(".atusdata.org")))


class SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parts = urlsplit(newurl)
        if parts.scheme != "https" or not is_ipums_host(parts.hostname):
            raise RuntimeError("Redirect left the approved IPUMS HTTPS hosts")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if parts.hostname != API_HOST:
            redirected.remove_header("Authorization")
        return redirected


OPENER = build_opener(SafeRedirect())


def open_url(url, *, key=None, payload=None):
    parts = urlsplit(url)
    if parts.scheme != "https" or not is_ipums_host(parts.hostname) or parts.username:
        raise RuntimeError("Expected an IPUMS HTTPS URL without embedded credentials")
    headers = {"User-Agent": "BlinderWeissResearch/1.0"}
    if key:
        if parts.hostname != API_HOST:
            raise RuntimeError("Credentials may only be sent to api.ipums.org")
        headers["Authorization"] = key
    body = None
    if payload is not None:
        body = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    try:
        return OPENER.open(Request(url, data=body, headers=headers), timeout=60)
    except HTTPError as exc:
        raise RuntimeError(f"IPUMS request failed with HTTP {exc.code}; response body withheld") from None
    except URLError:
        raise RuntimeError("IPUMS network request failed; rerun a read command to check status") from None


def api_key():
    key = os.environ.get("IPUMS_API_KEY", "").strip()
    if key:
        return key
    path = Path(os.environ.get("IPUMS_API_KEY_FILE", "/tmp/ipums_api_key"))
    if path.is_file():
        if stat.S_IMODE(path.stat().st_mode) & 0o077:
            raise RuntimeError("IPUMS key file must be private (chmod 600)")
        key = path.read_text().strip()
        if key:
            return key
    raise RuntimeError("No IPUMS API key: configure IPUMS_API_KEY privately or /tmp/ipums_api_key (mode 600), with USA and ATUS registration")


def api(collection, number=None, payload=None):
    suffix = f"/{int(number)}" if number is not None else ""
    url = f"https://{API_HOST}/extracts{suffix}?" + urlencode({"collection": collection, "version": 2})
    with open_url(url, key=api_key(), payload=payload) as response:
        return json.load(response)


def extract_summary(response, collection):
    definition = response.get("extractDefinition", {})
    return {"collection": collection, "number": response.get("number"),
            "status": response.get("status"), "checked_at": now(),
            "samples": sorted(definition.get("samples", {})),
            "variables": sorted(definition.get("variables", {})),
            "data_format": definition.get("dataFormat"),
            "data_structure": definition.get("dataStructure"),
            "download_file_count": len(response.get("downloadLinks", {}))}


def save_download(url, destination, *, key=None, expected_bytes=None, expected_sha256=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / ".staging"
    staging.mkdir(exist_ok=True)
    temporary = None
    digest = hashlib.sha256()
    size = 0
    try:
        with open_url(url, key=key) as response, tempfile.NamedTemporaryFile(dir=staging, delete=False) as stream:
            temporary = Path(stream.name)
            content_type = response.headers.get("Content-Type", "")
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        if expected_bytes is not None and size != int(expected_bytes):
            raise RuntimeError("Downloaded file size differs from IPUMS metadata")
        if expected_sha256 and digest.hexdigest() != expected_sha256.lower():
            raise RuntimeError("Downloaded file checksum differs from IPUMS metadata")
        if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() != digest.hexdigest():
            raise RuntimeError("Existing completed file differs; use a new output directory")
        temporary.replace(destination)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()
    return {"file": destination.name, "url": public_url(url), "downloaded_at": now(),
            "bytes": size, "sha256": digest.hexdigest(), "content_type": content_type,
            "ipums_sha256": expected_sha256, "ipums_bytes": expected_bytes}


def download_extract(collection, number, output):
    response = api(collection, number)
    summary = extract_summary(response, collection)
    if summary["status"] not in {"produced", "completed"}:
        print(json.dumps(summary, indent=2))
        raise RuntimeError("Extract is not complete; poll this same extract number, do not resubmit")
    links = response.get("downloadLinks", {})
    if not any(re.search(r"\.(dat|csv|dta|sav|sas7bdat)(\.gz|\.zip)?$", urlsplit(v.get("url", "")).path) for v in links.values()):
        raise RuntimeError("No microdata download link is available yet; poll this same extract number")
    destination = output / "raw" / f"{collection}_{int(number):05d}"
    manifest = {"extract": summary, "microdata_downloaded": False, "files": []}
    key = api_key()
    used_names = set()
    for label, link in links.items():
        url = link["url"]
        name = Path(urlsplit(url).path).name
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in {".", ".."} or name in used_names:
            raise RuntimeError("Unexpected or duplicate IPUMS download filename")
        used_names.add(name)
        entry = save_download(url, destination / name,
                              key=key if urlsplit(url).hostname == API_HOST else None,
                              expected_bytes=link.get("bytes"), expected_sha256=link.get("sha256"))
        entry["kind"] = label
        manifest["files"].append(entry)
        print(f"Downloaded {collection} extract {number}: {name} ({entry['bytes']} bytes)")
    manifest["microdata_downloaded"] = True
    atomic_json(destination / "manifest.json", manifest)
    return manifest


def fetch_metadata(output):
    # HTML parsing is optional; authenticated downloads use only the Python standard library.
    from bs4 import BeautifulSoup
    sources = json.loads((output / "metadata_sources.json").read_text())
    records = []
    samples = {}
    for source in sources:
        url, relative = source["url"], source["file"]
        with open_url(url) as response:
            raw = response.read()
        if source.get("format") == "yaml":
            content = raw
        else:
            soup = BeautifulSoup(raw, "html.parser")
            if source.get("kind") == "samples":
                samples[source["collection"]] = sorted({x.get("value") for x in soup.select('input[name="selectedSamples[]"]') if x.get("value")})
            expected = source.get("variable")
            if expected:
                headings = soup.find_all(["h1", "h2"])
                if not any(node.get_text(strip=True) == expected for node in headings):
                    raise RuntimeError(f"Variable documentation is missing for {expected}")
            for node in soup(["script", "style", "input", "meta"]):
                node.decompose()
            text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
            content = (text + "\n").encode()
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = path.parent / ".staging"
        staging.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=staging, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        temporary.replace(path)
        records.append({**source, "downloaded_at": now(), "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "source_response_sha256": hashlib.sha256(raw).hexdigest(),
                        "representation": "original YAML" if source.get("format") == "yaml" else "visible HTML text; scripts, inputs and metadata tags removed"})
    atomic_json(output / "metadata/available_samples.json", samples)
    atomic_json(output / "metadata/manifest.json", {"downloaded_at": now(), "microdata_downloaded": False, "files": records})
    return {"metadata_files": len(records), "microdata_downloaded": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("metadata", help="Download public documentation; requires beautifulsoup4")
    for name in ["list", "status", "download", "submit"]:
        command = commands.add_parser(name)
        command.add_argument("--collection", choices=["usa", "atus"], required=True)
        if name in {"status", "download"}:
            command.add_argument("--number", type=int, required=True)
        if name == "submit":
            command.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "metadata":
            print(json.dumps(fetch_metadata(args.output), indent=2))
        elif args.command == "download":
            manifest = download_extract(args.collection, args.number, args.output)
            print(json.dumps({"microdata_downloaded": True, "files": len(manifest["files"])}))
        elif args.command == "submit":
            spec = json.loads(args.spec.read_text())
            response = api(args.collection, payload=spec)
            summary = extract_summary(response, args.collection)
            # API results can contain authenticated download URLs: persist only this allowlist.
            atomic_json(args.output / "requests" / f"{args.collection}_{summary['number']}.json", {"request": spec, "result": summary})
            print(json.dumps(summary, indent=2))
        elif args.command == "status":
            print(json.dumps(extract_summary(api(args.collection, args.number), args.collection), indent=2))
        else:
            response = api(args.collection)
            extracts = response.get("data", response.get("extracts", [])) if isinstance(response, dict) else response
            print(json.dumps([extract_summary(item, args.collection) for item in extracts], indent=2))
    except (RuntimeError, OSError, ValueError) as exc:
        # Never print HTTP bodies or traceback request objects that may contain credentials.
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
