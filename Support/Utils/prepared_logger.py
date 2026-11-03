"""Save one completed prepared-data prediction or evaluation run.

Each call creates a new directory with config.yaml, metadata.json, either
predictions.csv or final_metrics.json, and run_summary.json written last. The
summary marks successful completion; an I/O failure may leave an incomplete
directory without it. Existing runs are never overwritten or reused.

This logger records supplied results and source declarations. It does not run
inference, calculate metrics, load source files, save model weights, or train.
The model fingerprint identifies the file originally read by its loader. Data
paths and contracts are recorded without rehashing arrays or proving provenance.
Callers must keep the model, data and result unchanged between execution and
logging; this helper cannot prove how a prediction tensor was produced.
"""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from uuid import uuid4

import torch
from torch import nn
from torch.utils.data import TensorDataset
import yaml

from support.data.contract import ModelDataContract
from support.data.manifest import PreparedDataManifest
from support.data.prepared import PreparedData
from support.evaluation.predict import PredictionResult
from support.evaluation.prepared import PreparedEvaluationResult
from support.models.artifact import ModelArtifact
from support.utils.prepared_config import validate_prepared_config


def _check_predictions(result, sample_count, num_classes):
    ids, probabilities = result.class_ids, result.probabilities
    if (not isinstance(ids, torch.Tensor) or ids.dtype != torch.int64
            or ids.device.type != 'cpu' or ids.shape != (sample_count,)):
        raise ValueError('Prediction class IDs must be a CPU int64 vector matching the dataset')
    if ((ids < 0) | (ids >= num_classes)).any():
        raise ValueError('Prediction class IDs are outside the configured class range')
    if (not isinstance(probabilities, torch.Tensor) or not probabilities.is_floating_point()
            or probabilities.device.type != 'cpu'
            or probabilities.shape != (sample_count, num_classes)
            or not torch.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()):
        raise ValueError('Prediction probabilities must be finite CPU [samples, classes] values in [0, 1]')
    sums = probabilities.to(torch.float64).sum(dim=1)
    if not torch.allclose(sums, torch.ones_like(sums), rtol=1e-5, atol=1e-6):
        raise ValueError('Prediction probability rows must sum approximately to one')
    # Keep class IDs selected from logits. Softmax rounding can introduce ties,
    # so recomputing IDs with probabilities.argmax() would change valid outputs.


def _check_evaluation(result, artifact, data, sample_count):
    if not data.has_labels:
        raise ValueError('Evaluation logging requires labeled prepared data')
    if type(result.sample_count) is not int or result.sample_count != sample_count:
        raise ValueError('Evaluation sample count does not match the supplied dataset')
    if (result.model_name != artifact.model_name
            or result.artifact_path != artifact.artifact_path
            or result.artifact_sha256 != artifact.artifact_sha256):
        raise ValueError('Evaluation model metadata does not match the supplied artifact')
    if (type(result.training_completed) is not bool
            or result.training_completed != artifact.training_completed
            or type(result.testing_only) is not bool
            or result.testing_only != (artifact.testing_only or data.manifest.testing_only)):
        raise ValueError('Evaluation purpose declarations do not match its inputs')
    if not isinstance(result.metrics, dict) or not result.metrics:
        raise ValueError('Evaluation metrics must be a nonempty dictionary')


def _json_text(value):
    """Serialize before creating a run directory; reject NaN and infinity."""
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'


def save_prepared_run(
    config: dict,
    artifact: ModelArtifact,
    data: PreparedData,
    result: PredictionResult | PreparedEvaluationResult,
    *,
    metadata: dict,
) -> Path:
    """Validate and save a completed result; return its new output directory.

    Args:
        config: Prepared configuration with absolute paths, normally returned
            by load_prepared_config(). Its mode selects the result type.
        artifact: The model artifact supplied for this run.
        data: The prepared dataset supplied for this run.
        result: Output of predict_prepared_data() or evaluate_prepared_data().
        metadata: JSON-compatible execution environment, including seed, Python
            and PyTorch versions, device and deterministic setting. Seed, device
            and determinism must agree with config. Additional fields such as
            library versions, timestamps and Git state are retained.

    Model/data testing-only flags are combined with logical OR. An optional
    metadata.testing_only flag can only retain or add testing-only status.
    These producer declarations do not establish research validity. Caller
    metadata is nested under 'environment' and cannot replace source records.

    Prediction CSV uses zero-based row positions in the prepared dataset's
    original order. probability_i corresponds to contract.class_names[i]; scores
    are softmax outputs, not calibrated correctness estimates. Raw features and
    target labels are not exported. Evaluation metrics are recorded as supplied.

    All input checks and JSON/YAML serialization precede filesystem writes.
    Exclusive file creation and a unique directory prevent silent overwrites.
    This is a single-call writer, not an append/resume or failure-recovery API.

    Raises:
        TypeError: Object types or metadata serialization are unsupported.
        ValueError: Config, metadata, purpose, contracts, paths or results differ,
            or exported numbers are invalid/nonfinite.
        OSError: Output creation or writing fails, including a directory collision.
    """
    config = validate_prepared_config(config)
    if not isinstance(artifact, ModelArtifact) or not isinstance(artifact.model, nn.Module):
        raise TypeError('artifact must be a ModelArtifact containing a torch.nn.Module')
    if not isinstance(artifact.contract, ModelDataContract):
        raise TypeError('artifact.contract must be a ModelDataContract')
    if (not isinstance(data, PreparedData) or not isinstance(data.dataset, TensorDataset)
            or not isinstance(data.manifest, PreparedDataManifest)):
        raise TypeError('data must contain a prepared manifest and TensorDataset')
    if len(data.dataset.tensors) not in (1, 2) or not len(data.dataset):
        raise ValueError('Logging requires a nonempty prepared dataset')
    if not isinstance(metadata, dict):
        raise TypeError('metadata must be an execution-environment dictionary')
    required_metadata = {'seed', 'python', 'pytorch', 'device', 'deterministic'}
    if not required_metadata.issubset(metadata):
        raise ValueError('Execution metadata must include seed, versions, device and determinism')
    experiment = config['experiment']
    if (type(metadata['seed']) is not int or metadata['seed'] != experiment['seed']
            or metadata['device'] != experiment['device']
            or type(metadata['deterministic']) is not bool
            or metadata['deterministic'] != experiment['deterministic']):
        raise ValueError('Execution metadata does not match the configured seed, device or determinism')
    if any(not isinstance(metadata[field], str) or not metadata[field].strip()
           for field in ('python', 'pytorch')):
        raise ValueError('Execution version declarations must be nonblank strings')
    extra_testing = metadata.get('testing_only', False)
    flags = (artifact.training_completed, artifact.testing_only, data.manifest.testing_only, extra_testing)
    if any(type(flag) is not bool for flag in flags):
        raise ValueError('Purpose declarations must be explicit booleans')
    if not artifact.training_completed and not artifact.testing_only:
        raise ValueError('An untrained model artifact must be marked testing_only')
    if (not isinstance(artifact.artifact_sha256, str)
            or re.fullmatch(r'[0-9a-f]{64}', artifact.artifact_sha256) is None):
        raise ValueError('The model artifact must have its loader-provided SHA-256')
    artifact.contract.assert_compatible(data.manifest.contract)

    for section, field in (
        ('experiment', 'output_dir'), ('model', 'artifact_path'), ('dataset', 'manifest_path'),
    ):
        if not Path(config[section][field]).is_absolute():
            raise ValueError('Use prepared configuration with absolute paths from load_prepared_config()')
    if Path(config['model']['artifact_path']).resolve() != artifact.artifact_path.resolve():
        raise ValueError('Configured model path does not match the supplied artifact')
    if Path(config['dataset']['manifest_path']).resolve() != data.manifest.manifest_path.resolve():
        raise ValueError('Configured manifest path does not match the supplied data')
    if config['dataset']['expected_role'] != data.manifest.role:
        raise ValueError('Configured dataset role does not match the supplied manifest')
    prediction_mode = config['mode'] == 'predict'
    expected_type = PredictionResult if prediction_mode else PreparedEvaluationResult
    if not isinstance(result, expected_type):
        raise TypeError('Result type does not match the configured prepared mode')
    if result.manifest != data.manifest:
        raise ValueError('Result manifest does not match the supplied dataset')
    sample_count = len(data.dataset)
    if prediction_mode:
        _check_predictions(result, sample_count, artifact.contract.num_classes)
    else:
        _check_evaluation(result, artifact, data, sample_count)

    testing_only = artifact.testing_only or data.manifest.testing_only or extra_testing
    created = datetime.now(timezone.utc)
    source = data.manifest
    record = {
        'schema_version': 1,
        'mode': config['mode'],
        'created_at_utc': created.isoformat(),
        'testing_only': testing_only,
        'environment': metadata,
        'model': {
            'model_name': artifact.model_name,
            'artifact_path': str(artifact.artifact_path),
            'artifact_sha256': artifact.artifact_sha256,
            'training_completed': artifact.training_completed,
            'testing_only': artifact.testing_only,
            'contract': artifact.contract.to_dict(),
        },
        'dataset': {
            'manifest_path': str(source.manifest_path),
            'dataset_name': source.dataset_name,
            'partition_name': source.partition_name,
            'role': source.role,
            'client_id': source.client_id,
            'data_format': source.data_format,
            'data_path': str(source.data_path),
            'preprocessing_path': str(source.preprocessing_path) if source.preprocessing_path is not None else None,
            'testing_only': source.testing_only,
            'has_labels': data.has_labels,
            'sample_count': sample_count,
            'contract': source.contract.to_dict(),
        },
    }
    output_file = 'predictions.csv' if prediction_mode else 'final_metrics.json'
    summary = {
        'schema_version': 1, 'status': 'completed', 'mode': config['mode'],
        'testing_only': testing_only, 'sample_count': sample_count, 'output_file': output_file,
    }
    metrics_text = None
    if not prediction_mode:
        metrics_text = _json_text({
            'schema_version': 1, 'testing_only': testing_only,
            'training_completed': artifact.training_completed,
            'artifact_sha256': artifact.artifact_sha256,
            'dataset_role': source.role, 'sample_count': sample_count,
            'class_names': list(artifact.contract.class_names), 'metrics': result.metrics,
        })
    config_text = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    metadata_text, summary_text = _json_text(record), _json_text(summary)

    root = Path(experiment['output_dir'])
    stamp = created.strftime('%Y%m%dT%H%M%S%fZ')
    directory = root / f'{experiment["name"]}_{stamp}_{uuid4().hex[:8]}'
    root.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    for filename, text in (('config.yaml', config_text), ('metadata.json', metadata_text)):
        with (directory / filename).open('x', encoding='utf-8') as stream:
            stream.write(text)
    if prediction_mode:
        names = artifact.contract.class_names
        with (directory / output_file).open('x', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow([
                'row_index', 'class_id', 'class_name',
                *(f'probability_{index}' for index in range(len(names))), 'testing_only',
            ])
            for row in range(sample_count):
                class_id = result.class_ids[row].item()
                writer.writerow([
                    row, class_id, names[class_id], *result.probabilities[row].tolist(), testing_only,
                ])
    else:
        with (directory / output_file).open('x', encoding='utf-8') as stream:
            stream.write(metrics_text)
    with (directory / 'run_summary.json').open('x', encoding='utf-8') as stream:
        stream.write(summary_text)
    return directory
