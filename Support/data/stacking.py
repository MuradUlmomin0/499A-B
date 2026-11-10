"""Recover the supplied notebook's lost preprocessing from training only.

Exact medians use disk-backed feature columns; scaling uses bounded partial_fit.
This is a reconstruction with today's libraries, not proof of historical identity.
No resampling or model training occurs. This format is separate from MLP v1 data.
"""
from contextlib import ExitStack
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import resource
import shutil
import tempfile
import zipfile

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler


def sha256_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _check_hash(path, expected):
    if (not isinstance(expected, str) or len(expected) != 64
            or any(c not in '0123456789abcdef' for c in expected)
            or sha256_file(path) != expected):
        raise ValueError(f'SHA-256 mismatch: {path}')


def _chunk_size(value):
    if type(value) is not int or not 1 <= value <= 20000:
        raise ValueError('chunk_rows must be between 1 and 20000')


def _schema(archive, member):
    if archive.namelist().count(member) != 1:
        raise ValueError(f'Expected exactly one CSV member: {member}')
    with archive.open(member) as stream:
        header = next(csv.reader(io.TextIOWrapper(stream, encoding='utf-8-sig', newline='')))
    if len(set(header)) != len(header) or 'label' not in header:
        raise ValueError(f'Invalid CSV header: {member}')
    with archive.open(member) as stream:
        sample = pd.read_csv(stream, nrows=1000)
    features = [name for name in header if name != 'label']
    dtypes = {name: 'int32' for name in sample.select_dtypes(include=['int64']).columns
              if name != 'label'}
    dtypes.update({name: 'float32' for name in sample.select_dtypes(include=['float64']).columns
                   if name != 'label'})
    if not 1 <= len(features) <= 512 or set(dtypes) != set(features):
        raise ValueError(f'Expected numeric notebook features and a label column: {member}')
    matrix_dtype = np.result_type(*[np.dtype(dtypes[name]) for name in features])
    # SimpleImputer converts integer-only matrices to float64.
    if matrix_dtype.kind != 'f':
        matrix_dtype = np.dtype('float64')
    return features, dtypes, matrix_dtype


def _impute(values, medians):
    if np.isinf(values).any():
        raise ValueError('Infinite values are unsupported by notebook preprocessing')
    rows, cols = np.where(np.isnan(values))
    values[rows, cols] = medians[cols]
    return values


def recover_preprocessing(archive, output, work_dir, expected_rows, chunk_rows,
                          archive_sha256, checkpoint_path, checkpoint_sha256,
                          notebook_path, notebook_sha256):
    """Write recovery.json + copied weights, never overwriting an existing folder.

    Inputs are pinned local files; expected_rows bounds disk use. Only train.csv
    is opened. Feature/class meanings come from that CSV and the notebook recipe,
    not from the unrelated 39-feature pilot. A hash marker makes interrupted or
    accidentally edited metadata fail loading. It is not a provenance signature.
    """
    output, work_dir, archive = Path(output), Path(work_dir), Path(archive)
    if output.exists():
        raise FileExistsError(output)
    _chunk_size(chunk_rows)
    if type(expected_rows) is not int or not 2 <= expected_rows <= 10000000:
        raise ValueError('expected_rows must be between 2 and 10000000')
    _check_hash(archive, archive_sha256)
    _check_hash(checkpoint_path, checkpoint_sha256)
    _check_hash(notebook_path, notebook_sha256)
    with zipfile.ZipFile(archive) as source:
        features, dtypes, matrix_dtype = _schema(source, 'Dataset/train.csv')
    work_dir.mkdir(parents=True, exist_ok=True)
    needed = expected_rows * len(features) * matrix_dtype.itemsize + 128 * 1024**2
    if shutil.disk_usage(work_dir).free < needed:
        raise OSError(f'Insufficient scratch disk; need {needed} bytes')
    output.mkdir(parents=True, exist_ok=False)
    try:
        with tempfile.TemporaryDirectory(prefix='stacking-recovery-', dir=work_dir) as scratch:
            columns = [Path(scratch)/f'{i}.bin' for i in range(len(features))]
            counts, rows = Counter(), 0
            with ExitStack() as stack, zipfile.ZipFile(archive) as source:
                writers = [stack.enter_context(path.open('wb')) for path in columns]
                stream = stack.enter_context(source.open('Dataset/train.csv'))
                for frame in pd.read_csv(stream, dtype=dtypes, chunksize=chunk_rows, low_memory=True):
                    values = frame[features].to_numpy(dtype=matrix_dtype)
                    if np.isinf(values).any():
                        raise ValueError('Infinite training values')
                    labels = frame['label']
                    if labels.isna().any() or any(not isinstance(x, str) or not x for x in labels):
                        raise ValueError('Expected nonempty string training labels')
                    counts.update(labels)
                    rows += len(frame)
                    if rows > expected_rows:
                        raise ValueError('Training row count exceeds expected row count')
                    for i, writer in enumerate(writers):
                        writer.write(values[:, i].tobytes())
            if rows != expected_rows:
                raise ValueError('Training row count differs from expected row count')
            classes = sorted(counts)
            if not 2 <= len(classes) <= 256:
                raise ValueError('Unsupported training class count')
            medians = np.empty(len(features), dtype=matrix_dtype)
            for i, path in enumerate(columns):
                mapped = np.memmap(path, dtype=matrix_dtype, mode='r', shape=(rows,))
                values = np.array(mapped)
                mapped._mmap.close()
                if np.isnan(values).all():
                    raise ValueError(f'All-missing training feature: {features[i]}')
                medians[i] = np.nanmedian(values, overwrite_input=True)
                del values
            if not np.isfinite(medians).all():
                raise ValueError('Nonfinite recovered medians')
            scaler = StandardScaler()
            with ExitStack() as stack:
                readers = [stack.enter_context(path.open('rb')) for path in columns]
                for start in range(0, rows, chunk_rows):
                    count = min(chunk_rows, rows-start)
                    values = np.empty((count, len(features)), dtype=matrix_dtype)
                    for i, reader in enumerate(readers):
                        values[:, i] = np.fromfile(reader, dtype=matrix_dtype, count=count)
                    scaler.partial_fit(_impute(values, medians))
            if not all(np.isfinite(v).all() for v in (scaler.mean_, scaler.var_, scaler.scale_)):
                raise ValueError('Nonfinite recovered scaling parameters')
        # Verify sources again before publishing a completed artifact.
        _check_hash(archive, archive_sha256)
        _check_hash(checkpoint_path, checkpoint_sha256)
        _check_hash(notebook_path, notebook_sha256)
        shutil.copyfile(checkpoint_path, output/'checkpoint.pth')
        _check_hash(output/'checkpoint.pth', checkpoint_sha256)
        # Validate the supplied blueprint/weights before any completion marker.
        # Kept local so inspecting data schemas does not initialize PyTorch.
        from support.models.stacking import load_stacking_checkpoint
        load_stacking_checkpoint(output/'checkpoint.pth', len(features), len(classes))
        description = {
            'schema_version': 1, 'architecture': 'notebook_stacking_v1',
            'status': 'reconstructed_unverified', 'testing_only': True,
            'historical_preprocessing_verified': False, 'split_provenance_verified': False,
            'training_rows': rows, 'feature_names': features, 'class_names': classes,
            'class_counts': dict(sorted(counts.items())), 'read_dtypes': dtypes,
            'matrix_dtype': str(matrix_dtype), 'medians': medians.tolist(),
            'mean': scaler.mean_.tolist(), 'var': scaler.var_.tolist(),
            'scale': scaler.scale_.tolist(), 'archive_sha256': archive_sha256,
            'checkpoint_sha256': checkpoint_sha256, 'notebook_sha256': notebook_sha256,
            'training_member': 'Dataset/train.csv', 'chunk_rows': chunk_rows,
            'method': 'training-only exact medians; chunked StandardScaler.partial_fit; sorted string LabelEncoder classes; no resampling',
            'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                         'pandas': pd.__version__, 'scikit_learn': sklearn.__version__},
            'peak_process_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
        payload = json.dumps(description, indent=2, allow_nan=False)+'\n'
        (output/'recovery.json').write_text(payload, encoding='utf-8')
        (output/'recovery.sha256').write_text(hashlib.sha256(payload.encode()).hexdigest()+'\n')
    except Exception as error:
        # Incomplete directories remain inspectable; no completion marker survives.
        (output/'recovery.sha256').unlink(missing_ok=True)
        (output/'recovery.json').unlink(missing_ok=True)
        (output/'failure.json').write_text(json.dumps({'status': 'failed', 'error': str(error)}))
        raise
    return output


def load_recovery(path):
    """Validate the diagnostic metadata and its copied checkpoint's integrity."""
    path = Path(path)
    if (path/'recovery.json').stat().st_size > 1024**2:
        raise ValueError('Recovery metadata exceeds 1 MiB')
    payload = (path/'recovery.json').read_bytes()
    if hashlib.sha256(payload).hexdigest() != (path/'recovery.sha256').read_text().strip():
        raise ValueError('Recovery metadata hash mismatch')
    r = json.loads(payload)
    if (r.get('schema_version') != 1 or r.get('architecture') != 'notebook_stacking_v1'
            or r.get('status') != 'reconstructed_unverified' or r.get('testing_only') is not True
            or r.get('historical_preprocessing_verified') is not False
            or r.get('split_provenance_verified') is not False):
        raise ValueError('Invalid diagnostic recovery declarations')
    features, classes = r['feature_names'], r['class_names']
    if (not isinstance(features, list) or not isinstance(classes, list)
            or not 1 <= len(features) <= 512 or not 2 <= len(classes) <= 256
            or any(not isinstance(v, str) or not v for v in features+classes)
            or len(set(features)) != len(features) or classes != sorted(set(classes))):
        raise ValueError('Invalid recovery feature/class names')
    if (r['matrix_dtype'] not in ('float32', 'float64')
            or set(r['read_dtypes']) != set(features)
            or any(v not in ('int32','float32') for v in r['read_dtypes'].values())
            or type(r['training_rows']) is not int or r['training_rows'] < 2):
        raise ValueError('Invalid recovery numeric schema')
    for name in ('medians', 'mean', 'var', 'scale'):
        values = np.asarray(r[name], dtype=np.float64)
        if values.shape != (len(features),) or not np.isfinite(values).all():
            raise ValueError(f'Invalid recovered {name}')
        if name == 'scale' and (values <= 0).any() or name == 'var' and (values < 0).any():
            raise ValueError(f'Invalid recovered {name}')
    _check_hash(path/'checkpoint.pth', r['checkpoint_sha256'])
    return r


def iter_stacking_data(archive, recovery, member='Dataset/validation.csv', chunk_rows=20000, limit=10000):
    """Yield bounded, transformed float32 features and encoded validation labels.

    Only validation is exposed here; this prevents accidental final-test scoring.
    Inference uses saved statistics, never partial_fit. The runner verifies the
    archive hash before and after execution. Labels map to sorted training names.
    """
    _chunk_size(chunk_rows)
    if member != 'Dataset/validation.csv':
        raise ValueError('Diagnostic inference accepts validation only')
    if type(limit) is not int or not 1 <= limit <= 100000:
        raise ValueError('Diagnostic validation limit must be between 1 and 100000')
    with zipfile.ZipFile(archive) as source:
        features, dtypes, matrix_dtype = _schema(source, member)
        if features != recovery['feature_names'] or str(matrix_dtype) != recovery['matrix_dtype']:
            raise ValueError('Validation feature order or matrix dtype differs from recovery')
        medians = np.asarray(recovery['medians'], dtype=matrix_dtype)
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(recovery['mean'], dtype=np.float64)
        scaler.var_ = np.asarray(recovery['var'], dtype=np.float64)
        scaler.scale_ = np.asarray(recovery['scale'], dtype=np.float64)
        scaler.n_features_in_ = len(features)
        scaler.n_samples_seen_ = np.int64(recovery['training_rows'])
        mapping = {name: i for i, name in enumerate(recovery['class_names'])}
        with source.open(member) as stream:
            for frame in pd.read_csv(stream, dtype=dtypes, chunksize=chunk_rows,
                                     nrows=limit, low_memory=True):
                labels = frame['label'].map(mapping)
                if labels.isna().any():
                    raise ValueError('Unknown or missing validation labels')
                values = frame[features].to_numpy(dtype=matrix_dtype, copy=True)
                values = scaler.transform(_impute(values, medians)).astype(np.float32)
                if not np.isfinite(values).all():
                    raise ValueError('Nonfinite transformed validation features')
                # pandas can return read-only arrays; torch.from_numpy requires
                # writable storage even when this runner only reads its labels.
                yield np.ascontiguousarray(values), labels.to_numpy(dtype=np.int64, copy=True)
