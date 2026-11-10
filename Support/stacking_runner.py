"""Explicit, separate recovery/validation workflow for the supplied neural ensemble.

No training, resampling, FL or final-test scoring. All outputs are diagnostic,
testing-only and unverified against the historical preprocessing objects.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
from time import perf_counter
from uuid import uuid4

import numpy as np
import torch
import yaml

from support.data.stacking import (
    iter_stacking_data, load_recovery, recover_preprocessing, sha256_file,
)
from support.evaluation.metrics import classification_metrics
from support.models.stacking import load_stacking_checkpoint
from support.utils.prepared_config import _PreparedConfigLoader
from support.utils.reproducibility import environment_metadata, seed_everything


REPOSITORY = Path(__file__).resolve().parents[1]


def _json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def load_stacking_config(path):
    """Validate exact explicit mode schemas; paths are relative to this YAML."""
    path = Path(path).resolve()
    with path.open(encoding='utf-8') as stream:
        config = yaml.load(stream, Loader=_PreparedConfigLoader)
    common = {'schema_version','mode','archive_path','archive_sha256','output_dir','chunk_rows'}
    extra = {
        'recover': {'checkpoint_path','checkpoint_sha256','notebook_path','notebook_sha256',
                    'work_dir','expected_rows'},
        'validate': {'recovery_dir','limit','batch_size','seed'},
    }
    if (not isinstance(config, dict) or not isinstance(config.get('mode'), str)
            or config['mode'] not in extra or set(config) != common | extra[config['mode']]
            or type(config['schema_version']) is not int or config['schema_version'] != 1):
        raise ValueError('Expected exact stacking recover/validate schema version 1')
    for name in ['archive_sha256'] + (
            ['checkpoint_sha256','notebook_sha256'] if config['mode']=='recover' else []):
        value = config[name]
        if not isinstance(value, str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError(f'Invalid SHA-256: {name}')
    bounds = {'chunk_rows': (1,20000)}
    bounds.update({'expected_rows': (2,10000000)} if config['mode']=='recover'
                  else {'limit': (1,100000), 'batch_size': (1,256), 'seed': (0,2**32-1)})
    for name, (low,high) in bounds.items():
        if type(config[name]) is not int or not low <= config[name] <= high:
            raise ValueError(f'{name} must be an integer between {low} and {high}')
    for name in list(config):
        if name.endswith('_path') or name.endswith('_dir'):
            if not isinstance(config[name], str) or not config[name].strip():
                raise ValueError(f'Invalid path: {name}')
            config[name] = str((path.parent/Path(config[name])).resolve())
    return config


def run_stacking(config_path):
    """Recover preprocessing or run a bounded CPU validation-prefix diagnostic."""
    config = load_stacking_config(config_path)
    started = perf_counter()
    if config['mode'] == 'recover':
        output = recover_preprocessing(
            archive=config['archive_path'], output=config['output_dir'],
            work_dir=config['work_dir'], expected_rows=config['expected_rows'],
            chunk_rows=config['chunk_rows'], archive_sha256=config['archive_sha256'],
            checkpoint_path=config['checkpoint_path'], checkpoint_sha256=config['checkpoint_sha256'],
            notebook_path=config['notebook_path'], notebook_sha256=config['notebook_sha256'])
        r = load_recovery(output)
        with (output/'config.yaml').open('x') as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
        _json(output/'run_summary.json', {
            'status': 'completed', 'mode': 'recover', 'testing_only': True,
            'historical_preprocessing_verified': False, 'training_rows': r['training_rows'],
            'input_dim': len(r['feature_names']), 'num_classes': len(r['class_names']),
            'model_state_compatible': True, 'duration_seconds': perf_counter()-started,
            'peak_process_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        })
        return output

    recovery_dir = Path(config['recovery_dir'])
    recovery = load_recovery(recovery_dir)
    recovery_hash = sha256_file(recovery_dir/'recovery.json')
    if (config['archive_sha256'] != recovery['archive_sha256']
            or sha256_file(config['archive_path']) != config['archive_sha256']):
        raise ValueError('SHA-256 archive identity differs from training recovery')
    model = load_stacking_checkpoint(recovery_dir/'checkpoint.pth',
                                    len(recovery['feature_names']), len(recovery['class_names']))
    seed_everything(config['seed'], True)
    torch.set_num_threads(1)
    output = Path(config['output_dir']) / (
        datetime.now(timezone.utc).strftime('stacking_diagnostic_%Y%m%dT%H%M%SZ_')+uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    try:
        with (output/'config.yaml').open('x') as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
        metadata = environment_metadata(config['seed'], 'cpu', True, REPOSITORY,
                                        testing_only=True, purpose='Reconstructed stacking diagnostic; NOT research results')
        metadata.update({
            'recovery_sha256': recovery_hash, 'archive_sha256': recovery['archive_sha256'],
            'checkpoint_sha256': recovery['checkpoint_sha256'], 'notebook_sha256': recovery['notebook_sha256'],
            'historical_preprocessing_verified': False, 'split_provenance_verified': False,
            'evaluation_role': 'validation', 'sample_selection': 'first configured rows; not representative',
            'class_names': recovery['class_names'], 'pytorch_threads': torch.get_num_threads(),
            'recovery_versions': recovery['versions'],
        })
        _json(output/'metadata.json', metadata)
        truth, predictions, loss_sum, rows = [], [], 0., 0
        with (output/'predictions.csv').open('x', newline='') as stream, torch.inference_mode():
            writer = csv.writer(stream)
            writer.writerow(['source_row_ordinal','true_class_id','true_label',
                             'predicted_class_id','predicted_label','max_probability'])
            for features, labels in iter_stacking_data(
                    config['archive_path'], recovery, chunk_rows=config['chunk_rows'], limit=config['limit']):
                for start in range(0,len(labels),config['batch_size']):
                    x = torch.from_numpy(features[start:start+config['batch_size']])
                    y = torch.from_numpy(labels[start:start+config['batch_size']])
                    logits = model(x)
                    if not torch.isfinite(logits).all():
                        raise ValueError('Nonfinite stacking model output')
                    confidence, pred = logits.softmax(dim=1).max(dim=1)
                    loss_sum += torch.nn.functional.cross_entropy(logits,y,reduction='sum').item()
                    for actual, predicted, probability in zip(y.tolist(), pred.tolist(), confidence.tolist()):
                        writer.writerow([rows,actual,recovery['class_names'][actual],
                                         predicted,recovery['class_names'][predicted],probability])
                        rows += 1
                    truth.extend(y.tolist()); predictions.extend(pred.tolist())
        if not rows:
            raise ValueError('Empty validation input')
        metrics = classification_metrics(np.asarray(truth),np.asarray(predictions),len(recovery['class_names']))
        metrics['loss'] = loss_sum/rows
        if (sha256_file(config['archive_path']) != recovery['archive_sha256']
                or sha256_file(recovery_dir/'recovery.json') != recovery_hash
                or sha256_file(recovery_dir/'checkpoint.pth') != recovery['checkpoint_sha256']):
            raise ValueError('Source or recovered artifact changed during execution')
        _json(output/'validation_metrics.json', {
            'testing_only': True, 'historical_preprocessing_verified': False,
            'evaluation_role': 'validation', 'class_names': recovery['class_names'], 'metrics': metrics})
        _json(output/'run_summary.json', {
            'status': 'completed', 'mode': 'validate', 'testing_only': True,
            'historical_preprocessing_verified': False, 'split_provenance_verified': False,
            'evaluation_role': 'validation', 'member': 'Dataset/validation.csv',
            'rows': rows, 'requested_row_limit': config['limit'],
            'selection': 'validation prefix; diagnostic only', 'research_results': False,
            'duration_seconds': perf_counter()-started,
            'peak_process_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    except Exception as error:
        _json(output/'failure.json', {'status': 'failed', 'error': str(error)})
        raise
    return output


def main():
    parser = argparse.ArgumentParser(description='Recover/validate notebook stacking model; diagnostic only, no training')
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    print(f'Stacking diagnostic artifacts: {run_stacking(args.config)}')


if __name__ == '__main__':
    main()
