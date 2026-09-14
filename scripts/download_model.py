#!/usr/bin/env python3
"""Download and validate an ONNX model before replacing its destination.

The receipt detects local corruption and URL changes; it is not an upstream
signature. Use --sha256 when a trusted publisher digest is available.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.error
import urllib.parse
import urllib.request


class HTTPSRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        require_https(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def require_https(url):
    if urllib.parse.urlsplit(url).scheme != "https":
        raise ValueError("model URLs and redirects must use HTTPS")


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            checksum.update(chunk)
    return checksum.hexdigest()


def validate_onnx(path):
    import onnx

    # Downloads must be self-contained. Never follow external tensor paths
    # embedded in a downloaded model.
    def check_embedded(message):
        if message.DESCRIPTOR.full_name == "onnx.TensorProto":
            if message.external_data or message.data_location == onnx.TensorProto.EXTERNAL:
                raise ValueError("downloaded ONNX must embed all tensor data")
        for field, value in message.ListFields():
            if field.message_type is not None:
                if field.is_repeated:
                    for child in value:
                        check_embedded(child)
                else:
                    check_embedded(value)

    try:
        model = onnx.load(str(path), load_external_data=False)
        check_embedded(model)
        onnx.checker.check_model(model)
    except Exception as error:
        raise ValueError(f"invalid ONNX: {error}") from error


def download(url, destination, mirror="", expected=""):
    require_https(url)
    if mirror:
        require_https(mirror)
    destination = Path(destination)
    receipt = destination.with_name(destination.name + ".download.json")
    try:
        saved = json.loads(receipt.read_text())
        if (isinstance(saved, dict) and destination.is_file()
                and saved.get("url") == url
                and saved.get("sha256") == digest(destination)
                and (not expected or saved["sha256"] == expected)):
            print(f"Verified existing model: {destination}")
            return
    except (OSError, ValueError):
        pass

    destination.parent.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(HTTPSRedirect())
    # A private directory on the destination filesystem enables atomic replace.
    with tempfile.TemporaryDirectory(prefix=".aji-download-", dir=destination.parent) as directory:
        temporary = Path(directory) / "model.onnx"
        last_error = None
        for source in (url, mirror):
            if not source:
                continue
            try:
                with opener.open(source, timeout=30) as response, temporary.open("wb") as output:
                    require_https(response.url)
                    size = 0
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        size += len(chunk)
                    length = response.headers.get("Content-Length")
                    if size == 0 or (length is not None and size != int(length)):
                        raise ValueError("empty or incomplete model download")
                actual = digest(temporary)
                if expected and actual != expected:
                    raise ValueError("model SHA256 mismatch")
                validate_onnx(temporary)
                metadata = Path(directory) / "receipt.json"
                metadata.write_text(json.dumps({"url": url, "sha256": actual}) + "\n")
                os.replace(temporary, destination)
                os.replace(metadata, receipt)
                print(f"Downloaded and checked: {destination}")
                return
            except (OSError, ValueError, urllib.error.URLError) as error:
                last_error = error
        raise RuntimeError(f"model download failed: {last_error}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--mirror", default="")
    parser.add_argument("--sha256", default="")
    args = parser.parse_args()
    if args.sha256 and (len(args.sha256) != 64 or any(c not in "0123456789abcdef" for c in args.sha256)):
        parser.error("--sha256 must contain 64 lowercase hexadecimal digits")
    download(args.url, args.destination, args.mirror, args.sha256)


if __name__ == "__main__":
    main()
