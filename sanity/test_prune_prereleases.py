"""Offline checks for the destructive prerelease retention policy."""

import contextlib
import importlib.util
import io
from pathlib import Path
import unittest
from urllib.error import HTTPError

spec = importlib.util.spec_from_file_location(
    "prune_prereleases", Path(__file__).resolve().parents[1] / "scripts/prune-prereleases.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def release(number, prerelease=True, draft=False):
    return dict(id=number, tag_name=f"test/{number}", prerelease=prerelease,
                draft=draft, published_at=f"2026-09-{number:02d}T00:00:00Z")


class FakeAPI:
    def __init__(self, releases, changed=None, missing_tag=False):
        self.releases = releases
        self.changed = changed
        self.missing_tag = missing_tag
        self.deleted = []

    def __call__(self, method, path):
        if method == "DELETE":
            self.deleted.append(path)
            if self.missing_tag and "/git/refs/" in path:
                raise HTTPError(path, 404, "Not Found", None, None)
            return
        if "?per_page=" in path:
            page = int(path.rsplit("=", 1)[1])
            return self.releases[(page - 1) * 100:page * 100]
        number = int(path.rsplit("/", 1)[1])
        return self.changed or next(r for r in self.releases if r["id"] == number)


class PruneTests(unittest.TestCase):
    def prune(self, api, apply=True):
        with contextlib.redirect_stdout(io.StringIO()):
            module.prune(api, "owner/repo", apply)

    def test_preserves_latest_stable_and_draft(self):
        api = FakeAPI([release(3, False), release(1), release(4, draft=True), release(2)])
        self.prune(api)
        self.assertEqual(api.deleted, ["/repos/owner/repo/releases/1",
                                       "/repos/owner/repo/git/refs/tags/test%2F1"])

    def test_preview_does_not_delete(self):
        api = FakeAPI([release(1), release(2)])
        self.prune(api, apply=False)
        self.assertEqual(api.deleted, [])

    def test_promoted_release_is_not_deleted(self):
        api = FakeAPI([release(1), release(2)], changed=release(1, False))
        with self.assertRaises(RuntimeError):
            self.prune(api)
        self.assertEqual(api.deleted, [])

    def test_pagination_finds_latest_after_first_page(self):
        api = FakeAPI([dict(release(1, False), id=100 + i, tag_name=f"stable-{i}")
                       for i in range(100)]
                      + [release(2), release(1)])
        self.prune(api)
        self.assertEqual(len(api.deleted), 2)
        self.assertTrue(api.deleted[0].endswith("/1"))

    def test_no_old_prereleases(self):
        for releases in ([], [release(1, False)], [release(1)]):
            api = FakeAPI(releases)
            self.prune(api)
            self.assertEqual(api.deleted, [])

    def test_already_missing_tag(self):
        api = FakeAPI([release(1), release(2)], missing_tag=True)
        self.prune(api)
        self.assertEqual(len(api.deleted), 2)

    def test_shared_tag_is_preserved(self):
        api = FakeAPI([release(1), release(2), dict(release(3, False), tag_name="test/1")])
        with self.assertRaises(RuntimeError):
            self.prune(api)
        self.assertEqual(api.deleted, [])


if __name__ == "__main__":
    unittest.main()
