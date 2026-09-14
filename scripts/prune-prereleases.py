#!/usr/bin/env python3
"""Keep the newest published prerelease; preview unless --apply is supplied."""

import argparse
import json
import os
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


def request(method, path):
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    api = os.environ.get("GITHUB_API_URL", "https://api.github.com")
    req = Request(f"{api}{path}", headers=headers, method=method)
    with urlopen(req, timeout=60) as response:
        body = response.read()
        return json.loads(body) if body else None


def list_releases(api, root):
    releases = []
    page = 1
    while True:
        batch = api("GET", f"{root}/releases?per_page=100&page={page}")
        releases.extend(batch)
        if len(batch) < 100:
            return releases
        page += 1


def prune(api, repository, apply=False):
    root = f"/repos/{repository}"
    releases = list_releases(api, root)
    candidates = sorted(
        (r for r in releases if r["prerelease"] and not r["draft"]
         and r.get("published_at")),
        key=lambda r: (r["published_at"], r["id"]), reverse=True,
    )
    if not candidates:
        print("No published prereleases to prune.")
        return
    keep = candidates[0]
    print(f"Keeping {keep['tag_name']} ({keep['id']})", flush=True)
    for old in candidates[1:]:
        release_path = f"{root}/releases/{old['id']}"
        # Recheck each candidate in case it was promoted since the list call.
        current = api("GET", release_path)
        if (not current["prerelease"] or current["draft"]
                or current["tag_name"] != old["tag_name"]
                or current.get("published_at") != old["published_at"]):
            raise RuntimeError(f"Release changed; refusing to delete {old['tag_name']}")
        # Do not delete a tag also referenced by a release we are keeping.
        if any(r["id"] != old["id"] and r["tag_name"] == old["tag_name"]
               for r in releases):
            raise RuntimeError(f"Shared release tag: {old['tag_name']}")
        print(f"{'Deleting' if apply else 'Would delete'} release and tag "
              f"{old['tag_name']} ({old['id']})", flush=True)
        if apply:
            api("DELETE", release_path)
            tag = quote(old["tag_name"], safe="")
            try:
                api("DELETE", f"{root}/git/refs/tags/{tag}")
            except HTTPError as error:
                error.close()
                if error.code != 404:
                    raise
                print(f"Tag already absent: {old['tag_name']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="delete old releases and tags")
    args = parser.parse_args()
    repository = os.environ["GITHUB_REPOSITORY"]
    if args.apply and not os.environ.get("GH_TOKEN"):
        parser.error("--apply requires GH_TOKEN with repository contents:write")
    prune(request, repository, args.apply)
