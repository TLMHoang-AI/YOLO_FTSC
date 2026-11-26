"""Rate-aware persistence policy layered on the existing HFResultArchive protocol."""
from __future__ import annotations
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import time
from utils.hf_result_archive import HFResultArchive


def retry_after_seconds(error, now=None):
    response = getattr(error, 'response', None)
    headers = {str(k).lower(): str(v) for k, v in getattr(response, 'headers', {}).items()}
    now = time.time() if now is None else now
    value = headers.get('retry-after', '')
    if value:
        try:
            return max(1, float(value))
        except ValueError:
            try:
                return max(1, parsedate_to_datetime(value).timestamp()-now)
            except (ValueError, TypeError, OverflowError):
                pass
    # HF uses a structured RateLimit field containing seconds until reset.
    match = re.search(r'(?:^|[;,\s])t=(\d+)', headers.get('ratelimit', ''))
    if match:
        return max(1, int(match.group(1)))
    reset = headers.get('x-ratelimit-reset', headers.get('ratelimit-reset', ''))
    if reset:
        try:
            number = float(reset)
            return max(1, number-now if number > now else number)
        except ValueError:
            pass
    return 3600  # conservative fallback for the hourly repository commit quota


class RateLimitedArchive(HFResultArchive):
    """Keep transactional upload/marker semantics; wait instead of hammering 429."""
    def _retry(self, operation):
        failures = 0
        while True:
            try:
                return operation()
            except Exception as error:
                response = getattr(error, 'response', None)
                if getattr(response, 'status_code', None) == 429:
                    seconds = retry_after_seconds(error)+2
                    deadline = time.monotonic()+seconds
                    print(f'HF_RATE_LIMIT: pausing writes for {seconds:.0f}s; saved results retained.', flush=True)
                    while time.monotonic() < deadline:
                        remaining = deadline-time.monotonic()
                        self._sleep(min(30, remaining))
                        if time.monotonic() < deadline:
                            print(f'HF_RATE_LIMIT: retry in {deadline-time.monotonic():.0f}s', flush=True)
                    continue
                failures += 1
                if failures >= self.retries:
                    raise
                self._sleep(self.retry_backoff_seconds*(2**(failures-1)))


def upload_report_bundle(store, names):
    """Publish all report files together and skip unchanged byte-identical bundles."""
    source = store.args.project/store.args.dataset
    fingerprint = {name: hashlib.sha256((source/name).read_bytes()).hexdigest() for name in names}
    saved = source/'hf_report_bundle_uploaded.json'
    identity = {'repo': store.args.repo, 'prefix': store.remote_prefix, 'files': fingerprint}
    if saved.is_file() and json.loads(saved.read_text()) == identity:
        print('HF_REPORTS_UNCHANGED: upload skipped', flush=True)
        return
    with tempfile.TemporaryDirectory(dir=store.args.project) as temp:
        for name in names:
            shutil.copyfile(source/name, Path(temp)/name)
        store.archive._retry(lambda: store.api.upload_folder(
            repo_id=store.args.repo, repo_type='dataset', folder_path=temp,
            path_in_repo=f'{store.remote_prefix}/reports',
            commit_message=f'update {store.args.dataset} INT8 reports (one commit)'))
    saved.write_text(json.dumps(identity, sort_keys=True)+'\n')
    print('HF_REPORTS_UPLOADED: one bundled commit', flush=True)
