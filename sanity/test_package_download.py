#!/usr/bin/env python3
"""Exercise the packager's download function without network or system changes."""
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
import urllib.parse


ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD = re.search(
    r'^download\(\) \{\n.*?^\}',
    (ROOT / 'scripts/package-ubuntu24-runtime.sh').read_text(),
    re.MULTILINE | re.DOTALL,
).group()


class PackageDownloadTests(unittest.TestCase):
    def run_download(self, url, failures=0, empty=False):
        with tempfile.TemporaryDirectory(prefix='aji-package-test-') as directory:
            root = Path(directory)
            destination = root / 'model archive.7z'
            destination.write_bytes(b'existing archive')
            script = DOWNLOAD + r'''
sleep() { printf '%s\n' "$1" >> "$TEST_DIR/delays"; }
curl() {
    local output='' previous='' arg
    for arg; do
        if [[ "$previous" == --output ]]; then output="$arg"; fi
        previous="$arg"
    done
    printf '%s\n' "${@: -1}" >> "$TEST_DIR/urls"
    calls=$((calls + 1))
    if ((calls <= FAILURES)); then
        printf partial > "$output"
        return 22
    fi
    if [[ "$EMPTY" == 1 ]]; then
        : > "$output"
    else
        printf complete > "$output"
    fi
}
calls=0
download "$URL" "$TEST_DIR/model archive.7z"
'''
            result = subprocess.run(
                ['bash', '--noprofile', '--norc', '-euo', 'pipefail', '-c', script],
                env={**os.environ, 'TEST_DIR': directory, 'URL': url,
                     'FAILURES': str(failures), 'EMPTY': str(int(empty))},
                capture_output=True, text=True, timeout=10,
            )
            urls = (root / 'urls').read_text().splitlines()
            delays = (root / 'delays').read_text().splitlines() if (root / 'delays').exists() else []
            self.assertFalse(Path(str(destination) + '.part').exists())
            return result, urls, delays, destination.read_bytes()

    def test_success_needs_no_retry(self):
        url = 'https://example.test/model.7z'
        result, urls, delays, data = self.run_download(url)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(urls, [url])
        self.assertEqual(delays, [])
        self.assertEqual(data, b'complete')

    def test_gateway_failures_refresh_github_url_and_back_off(self):
        url = 'https://github.com/owner/repo/releases/download/models/rife.7z?download=1'
        result, urls, delays, data = self.run_download(url, failures=2)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(urls[0], url)
        self.assertEqual(len(set(urls)), 3)
        for retry in urls[1:]:
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(retry).query)
            self.assertEqual(query['download'], ['1'])
            self.assertIn('aji_retry', query)
        self.assertEqual(delays, ['5', '10'])
        self.assertEqual(data, b'complete')

    def test_exhausted_retries_preserve_existing_archive(self):
        url = 'https://example.test/model.7z?signature=unchanged'
        result, urls, delays, data = self.run_download(url, failures=5)
        self.assertEqual(result.returncode, 22)
        self.assertEqual(urls, [url] * 5)
        self.assertEqual(delays, ['5', '10', '20', '40'])
        self.assertEqual(data, b'existing archive')
        self.assertIn(url, result.stderr)

    def test_empty_response_cannot_replace_existing_archive(self):
        result, urls, _, data = self.run_download('https://example.test/model.7z', empty=True)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(len(urls), 5)
        self.assertEqual(data, b'existing archive')


if __name__ == '__main__':
    unittest.main()
