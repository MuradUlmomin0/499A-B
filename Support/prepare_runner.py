"""Memory-bounded diagnostic preparation; no model execution or training."""
import argparse
import json
from pathlib import Path
import resource
import shutil
import tempfile
from time import perf_counter
import yaml
from Support.data.pilot_sampling import audit_and_sample
from Support.data.pilot_export import export_pilot
from Support.data.split_checks import file_sha256
from Support.Utils.preparation_config import load_preparation_config


def _existing_parent(path):
    path = Path(path)
    while not path.exists():
        path = path.parent
    return path


def _json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def run_preparation(config_path):
    """Prepare a bounded real-data diagnostic pilot, never optimizing a model.

    Source hash/config/disk checks precede output creation. Owned scratch is
    removed; failures retain diagnostics without a completed summary. Existing
    output directories are not replaced. Inputs must remain stable throughout.
    """
    started = perf_counter()
    config = load_preparation_config(config_path)
    source = Path(config['source']['archive_path'])
    output, work_root = map(Path, [config['output']['directory'], config['output']['work_dir']])
    if output.exists():
        raise FileExistsError(f'Output already exists; choose a new directory: {output}')
    before = source.stat()
    if file_sha256(source) != config['source']['sha256']:
        raise ValueError('Archive SHA-256 differs from the audited source')
    if shutil.which('sort') is None:
        raise RuntimeError('Coreutils sort is required for the disk-backed audit')
    if work_root.exists() and not work_root.is_dir():
        raise ValueError('Scratch root must be a directory')
    valid_rows = sum(config['source']['expected_valid_counts'].values())
    pilot_bytes = config['pilot']['max_rows'] * (len(config['source']['feature_names'])*4 + 256)
    required_work = valid_rows * 67 * 3 + pilot_bytes * 2 + 256*1024**2
    if shutil.disk_usage(_existing_parent(work_root)).free < required_work:
        raise ValueError(f'Insufficient scratch disk space: require approximately {required_work:,} bytes')
    if shutil.disk_usage(_existing_parent(output)).free < pilot_bytes*3 + 256*1024**2:
        raise ValueError('Insufficient output disk space')
    meminfo = Path('/proc/meminfo')
    available = None
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith('MemAvailable:'):
                available = int(line.split()[1])*1024
        if available is not None and available < 1024**3:
            raise ValueError('Less than 1 GiB RAM is available; close other applications before preparing')
    output.mkdir(parents=True, exist_ok=False)
    work_root.mkdir(parents=True, exist_ok=True)
    (output/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    def progress(value):
        print(json.dumps({**value, 'elapsed_seconds': round(perf_counter()-started, 1)}), flush=True)
    try:
        with tempfile.TemporaryDirectory(prefix='ciciot_pilot_', dir=work_root) as temporary:
            scratch = Path(temporary)
            try:
                sample, audit = audit_and_sample(config, scratch, progress)
                _json(output/'duplicate_audit.json', audit)
                progress({'stage': 'split_scale_export', 'groups': len(sample['labels'])})
                exported = export_pilot(sample, config, output)
                _json(output/'preparation_report.json', exported)
                after = source.stat()
                if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns or file_sha256(source) != config['source']['sha256']:
                    raise ValueError('Source archive changed during preparation')
                summary = {'schema_version': 1, 'status': 'completed', 'workflow': 'diagnostic_data_preparation',
                    'testing_only': True, 'training_performed': False, 'simulated_clients': True,
                    'label_source': 'folder_name_unverified', 'provenance_verified': False,
                    'source_sha256': config['source']['sha256'], 'seed': config['pilot']['seed'],
                    'rows_exported': len(sample['labels']), 'federated_bundle': 'federated_bundle.json',
                    'all_role_record_ids_disjoint': exported['all_role_record_ids_disjoint'],
                    'all_role_feature_groups_disjoint': exported['all_role_feature_groups_disjoint'],
                    'memory': {'user_reported_system_ram_gb': 8, 'available_before_bytes': available,
                               'peak_process_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                               'peak_sort_rss_kib': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                               'sort_buffer_mib': 32, 'sort_address_space_limit_mib': 512,
                               'csv_chunk_rows': config['pilot']['chunk_rows']},
                    'duration_seconds': perf_counter()-started}
            finally:
                for name in ['quarantine.jsonl', 'conflicting_groups.jsonl']:
                    if (scratch/name).is_file():
                        shutil.copyfile(scratch/name, output/name)
        _json(output/'run_summary.json', summary)
        progress({'stage': 'completed', 'rows_exported': summary['rows_exported'], 'output': str(output)})
        return output
    except Exception as error:
        _json(output/'failure.json', {'status': 'failed', 'error': str(error), 'training_performed': False})
        raise


def main():
    """Low-memory standalone entry avoids importing the FL/model runners."""
    parser = argparse.ArgumentParser(description='Prepare a diagnostic CICIoT2023 pilot without training')
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    output = run_preparation(args.config)
    print(f'Data preparation completed (diagnostic only). Artifacts: {output}')


if __name__ == '__main__':
    main()
