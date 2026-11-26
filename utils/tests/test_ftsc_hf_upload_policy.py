from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock
import tempfile
from utils.ftsc_hf_upload_policy import RateLimitedArchive, retry_after_seconds, upload_report_bundle


def limited(headers):
    error=RuntimeError('429')
    error.response=SimpleNamespace(status_code=429,headers=headers)
    return error


class UploadPolicyTests(TestCase):
    def test_retry_after_and_structured_reset(self):
        self.assertEqual(retry_after_seconds(limited({'Retry-After':'120'})),120)
        self.assertEqual(retry_after_seconds(limited({'RateLimit':'"commit";r=0;t=900'})),900)
        self.assertEqual(retry_after_seconds(limited({})),3600)

    def test_http_date_retry_after(self):
        self.assertEqual(retry_after_seconds(limited({'retry-after':'Thu, 01 Jan 1970 00:02:00 GMT'}),now=60),60)

    def test_429_waits_and_retries_without_consuming_transient_budget(self):
        clock=[0.]
        def sleep(seconds):clock[0]+=seconds
        archive=RateLimitedArchive('owner/repo',token='test',sleep=sleep,retries=1)
        op=mock.Mock(side_effect=[limited({'retry-after':'60'}),'uploaded'])
        with mock.patch('utils.ftsc_hf_upload_policy.time.monotonic',side_effect=lambda:clock[0]):
            self.assertEqual(archive._retry(op),'uploaded')
        self.assertEqual(op.call_count,2)
        self.assertEqual(clock[0],62)

    def test_reports_use_one_folder_commit_and_skip_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);folder=root/'LEVIR-Ship';folder.mkdir()
            for name in ('summary.json','metrics.csv'):(folder/name).write_text(name)
            store=SimpleNamespace(args=SimpleNamespace(project=root,dataset='LEVIR-Ship',repo='owner/repo'),
                                  remote_prefix='LEVIR-Ship/test',api=mock.Mock(),archive=mock.Mock())
            store.archive._retry.side_effect=lambda operation:operation()
            upload_report_bundle(store,('summary.json','metrics.csv'))
            upload_report_bundle(store,('summary.json','metrics.csv'))
            self.assertEqual(store.api.upload_folder.call_count,1)
            self.assertEqual(store.api.upload_folder.call_args.kwargs['path_in_repo'],'LEVIR-Ship/test/reports')
            (folder/'summary.json').write_text('changed')
            upload_report_bundle(store,('summary.json','metrics.csv'))
            self.assertEqual(store.api.upload_folder.call_count,2)
