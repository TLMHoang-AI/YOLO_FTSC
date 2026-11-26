"""Best-only LEVIR V2 INT8 driver with HFResultArchive transactional persistence.

Keep the measured V1 evaluator unmodified: its source hashes remain valid for
existing engines/results. Archive failures stop the run before the next export.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

from utils import ftsc_hf_all_trt_int8_paper_v1 as core
from utils import ftsc_hf_all_trt_int8_levir_fp16parity_v2 as evaluator
from utils.hf_result_archive import HFResultArchive, MARKER_FILENAME, RUN_SUMMARY_FILENAME
from utils.ftsc_hf_upload_policy import RateLimitedArchive, upload_report_bundle
from utils.ftsc_legacy_detect_support import legacy_detect_support

DATASET = 'LEVIR-Ship'
VARIANT = 'int8_static_fp16parity_v2_callfix'
PREFIX = f'{DATASET}/{VARIANT}'


def best_inventory(inventory):
    selected = {**inventory, 'checkpoints': [r for r in inventory['checkpoints'] if r['kind'] == 'best.pt']}
    selected['count'] = len(selected['checkpoints'])
    return selected


def required_files(folder):
    return tuple(sorted(p.relative_to(folder).as_posix() for p in folder.rglob('*')
                        if p.is_file() and p.name not in (MARKER_FILENAME, RUN_SUMMARY_FILENAME)))


class Persistence:
    def __init__(self, args):
        from huggingface_hub import HfApi
        self.args = args
        self.audit_context = evaluator.dataset_context(args)[2:] if args.data_yaml else None
        self.api = HfApi(token=os.environ.get('HF_TOKEN'))
        self.remote_prefix = PREFIX
        self.archive = RateLimitedArchive(repo_id=args.repo, api=self.api,
                                       cache_dir=args.project/'hf_restore_cache', retries=3,
                                       retry_backoff_seconds=2)
        self.revision = self.api.repo_info(repo_id=args.repo, repo_type='dataset').sha
        self.files = set(self.api.list_repo_files(repo_id=args.repo, repo_type='dataset', revision=self.revision))

    def download(self, name):
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(repo_id=self.args.repo, repo_type='dataset',
                                   revision=self.revision, filename=name,
                                   token=os.environ.get('HF_TOKEN')))

    def inventory(self):
        local = evaluator.inventory_file(self.args.project)
        remote = f'{PREFIX}/hf_pinned_inventory.json'
        if remote in self.files:
            saved = json.loads(self.download(remote).read_text())
            evaluator.validate_inventory(saved, self.args.repo)
            if local.exists():
                core.need(json.loads(local.read_text()) == saved,
                          'Local and archived pinned inventories differ; refusing overwrite')
            else:
                core.atomic_json(local, saved)
        inventory = evaluator.inventory(self.args.project, self.args.repo)
        self.pinned = inventory
        if remote not in self.files:
            self.upload_file(local, remote)
        return inventory

    def upload_file(self, path, remote):
        self.archive._retry(lambda: self.api.upload_file(
            repo_id=self.args.repo, repo_type='dataset', path_or_fileobj=str(path),
            path_in_repo=remote, commit_message=f'persist LEVIR V2 INT8 {path.name}'))

    def restore(self, row):
        folder = core.job_dir(self.args.project, DATASET, row)
        if (folder/'metrics_int8.json').exists():
            return False
        prefix = f'{PREFIX}/{row["job_id"]}'
        marker_path = f'{prefix}/{MARKER_FILENAME}'
        if marker_path not in self.files:
            return False
        marker = json.loads(self.download(marker_path).read_text())
        required = tuple(p for p in marker['artifact_sha256'] if p != RUN_SUMMARY_FILENAME)
        from huggingface_hub import snapshot_download
        def scoped_snapshot(**kwargs):
            # Reuse the archive restore protocol, without downloading all LEVIR training runs.
            kwargs['allow_patterns'] = [f'{prefix}/**']
            kwargs['revision'] = self.revision
            return snapshot_download(**kwargs)
        restorer = HFResultArchive(repo_id=self.args.repo, api=self.api,
            cache_dir=self.args.project/'hf_restore_cache', snapshot_download_fn=scoped_snapshot)
        with tempfile.TemporaryDirectory(dir=self.args.project) as temp:
            result = restorer.restore_completed_runs(dataset_slug=DATASET, project=temp,
                variants=[VARIANT], seeds=[row['checkpoint_seed']],
                run_name_fn=lambda _v, _s: row['job_id'], required_artifacts=required,
                contract_file='metrics_int8.json')
            core.need(len(result['restored']) == 1, 'Marked HF archive was not restored')
            source = Path(result['restored'][0]['run_dir'])
            for file in source.rglob('*'):
                if file.is_file():
                    local = folder/file.relative_to(source)
                    core.need(not local.exists() or core.sha256(local) == core.sha256(file),
                              f'Refusing to overwrite differing partial local artifact: {local}')
            folder.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, folder, dirs_exist_ok=True)
        print(f'RESTORED {row["job_id"]} (SHA verified)', flush=True)
        return True

    def publish(self, row, hashes):
        folder = core.job_dir(self.args.project, DATASET, row)
        record = json.loads((folder/'metrics_int8.json').read_text())
        context = self.audit_context or (record['data_provenance'], record['train_calibration_fingerprint'])
        evaluator.verify_completed(folder, row, self.pinned, record, hashes, *context)
        if self.audit_context is None:
            self.audit_context = context
        required = required_files(folder)
        remote_marker = f'{PREFIX}/{row["job_id"]}/{MARKER_FILENAME}'
        if remote_marker in self.files:
            marker = json.loads(self.download(remote_marker).read_text())
            expected = {p: core.sha256(folder/p) for p in required}
            core.need(all(marker.get('artifact_sha256', {}).get(p) == sha for p, sha in expected.items()),
                      f'Remote completed archive differs: {row["job_id"]}')
            print(f'HF_VERIFIED {row["job_id"]}', flush=True)
            return
        result = self.archive.publish_run(dataset_slug=DATASET, variant=VARIANT,
            seed=row['checkpoint_seed'], run_name=row['job_id'], run_dir=folder,
            required_artifacts=required,
            summary_metadata={'contract': evaluator.CONTRACT, 'source': row['path'],
                              'hf_revision': record['hf_revision'], 'metrics': record['metrics']})
        print('HF_UPLOADED', json.dumps(result), flush=True)

    def reports(self, inventory, hashes):
        report = evaluator.aggregate(self.args, best_inventory(inventory), hashes,
                                     *(self.audit_context or (None, None)))
        upload_report_bundle(self, ('summary.json', 'all_checkpoint_status.csv',
                                    'measured_int8_metrics.csv', 'complete_three_seed_best_aggregate.csv'))
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('backup','run','aggregate'), default='backup')
    parser.add_argument('--repo', default=core.HF_REPO)
    parser.add_argument('--project', type=Path, default=evaluator.DEFAULT_PROJECT)
    parser.add_argument('--data-yaml', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--max-jobs', type=int, default=0)
    args = parser.parse_args(argv)
    args.dataset = DATASET
    args.project = args.project.resolve()
    args.project.mkdir(parents=True, exist_ok=True)
    core.need(args.start_index >= 0 and args.max_jobs >= 0, 'Invalid best-only selection')
    store = Persistence(args)
    inventory = store.inventory()
    hashes = evaluator.source_hashes()
    # Protect all completed work, including historical last.pt results, before new exports.
    completed = [r for r in inventory['checkpoints']
                 if (core.job_dir(args.project, DATASET, r)/'metrics_int8.json').exists()]
    for index, row in enumerate(completed):
        print(f'BACKUP {index+1}/{len(completed)} {row["job_id"]}', flush=True)
        store.publish(row, hashes)
    if args.phase == 'run':
        core.need(args.data_yaml is not None, 'Canonical LEVIR --data-yaml is required')
        context = evaluator.dataset_context(args)
        rows = best_inventory(inventory)['checkpoints'][args.start_index:]
        if args.max_jobs:
            rows = rows[:args.max_jobs]
        hardware = core.runtime_identity(args.device)
        for index, row in enumerate(rows):
            print(f'BEST_ONLY {index+1}/{len(rows)} {row["path"]}', flush=True)
            store.restore(row)
            if (core.job_dir(args.project, DATASET, row)/'metrics_int8.json').is_file():
                result = evaluator.evaluate_one(args, inventory, row, hashes, hardware, context)
            else:
                with legacy_detect_support(core.job_dir(args.project, DATASET, row), row):
                    result = evaluator.evaluate_one(args, inventory, row, hashes, hardware, context)
            print(json.dumps(result), flush=True)
            store.publish(row, hashes)  # upload failure propagates; never continue unprotected
            if result['status'] != 'REUSED_VERIFIED':
                store.reports(inventory, hashes)
            import gc, torch
            gc.collect()
            torch.cuda.empty_cache()
    report = store.reports(inventory, hashes)
    print(json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    main()
