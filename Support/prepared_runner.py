"""Run prepared-data prediction or evaluation using the existing components.

This callable coordinates configuration, compatible saved weights and prepared
arrays, reproducibility, inference/evaluation, and logging. It does not fit a
model, apply preprocessing, choose partitions, or run a federated algorithm.
Source declarations and successful execution do not establish research validity.
"""

from pathlib import Path
from time import perf_counter

import torch

from support.data.prepared import load_prepared_data
from support.evaluation.predict import predict_prepared_data
from support.evaluation.prepared import evaluate_prepared_data
from support.models.artifact import load_model_artifact
from support.utils.prepared_config import load_prepared_config
from support.utils.prepared_logger import save_prepared_run
from support.utils.reproducibility import environment_metadata, seed_everything


REPOSITORY = Path(__file__).resolve().parents[1]


def run_prepared(config_path: str | Path) -> Path:
    """Execute one configured prepared run and return its completed log folder.

    Args:
        config_path: Version 1 YAML accepted by load_prepared_config(). Its mode
            is 'predict' or 'evaluate'. Model/data paths are resolved by their
            existing loaders; feature counts and class meanings are not guessed.

    Prediction accepts labeled or unlabeled data. Evaluation requires labels,
    including when the explicitly requested dataset role is 'predict'. Role and
    contract checks precede execution. Test-role scores must not be used for
    checkpoint selection, aggregation choices, or other training decisions.

    The runner deliberately seeds process-wide Python, NumPy and PyTorch RNGs
    and sets deterministic execution through seed_everything(). These settings
    are not restored after the call. The loaded MLP is moved to the configured
    CPU/CUDA device; unavailable requested CUDA raises an error. Saved weights
    are used directly and no optimizer is constructed.

    Metadata retains testing-only status if either model or data declares it.
    execution_duration_seconds measures prediction or evaluation, including
    evaluation metric computation, while excluding loading, device placement,
    environment metadata collection and output writing. Thread count is recorded
    without changing it. Timestamps, timings and output-folder names naturally
    differ between runs; those differences do not imply different predictions.

    Input-loading and execution failures propagate before output writing begins.
    Logger I/O failures can leave an incomplete directory, as documented by
    save_prepared_run(). Missing files, invalid metadata/contracts/arrays and
    model execution errors retain their original loader/helper exceptions.

    This is a callable API. Connection to main.py is a separate integration step.
    """
    config_path = Path(config_path).resolve()
    config = load_prepared_config(config_path)
    experiment = config['experiment']
    if experiment['device'] == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested for prepared execution but unavailable')

    artifact = load_model_artifact(config['model']['artifact_path'])
    data = load_prepared_data(
        config['dataset']['manifest_path'],
        expected_role=config['dataset']['expected_role'],
        expected_contract=artifact.contract,
    )
    if config['mode'] == 'evaluate' and not data.has_labels:
        raise ValueError('Prepared evaluation requires labeled data')

    # Inputs are validated and saved weights are loaded before execution starts
    # from the configured RNG state.
    seed_everything(experiment['seed'], experiment['deterministic'])
    artifact.model.to(experiment['device'])
    testing_only = artifact.testing_only or data.manifest.testing_only
    operation = 'prediction' if config['mode'] == 'predict' else 'evaluation'
    purpose = f'Prepared-data {operation}'
    if testing_only:
        purpose += '; testing-only verification'
    metadata = environment_metadata(
        experiment['seed'], experiment['device'], experiment['deterministic'], REPOSITORY,
        testing_only=testing_only, purpose=purpose,
    )
    metadata['config_path'] = str(config_path)
    metadata['pytorch_threads'] = torch.get_num_threads()

    started = perf_counter()
    if config['mode'] == 'predict':
        result = predict_prepared_data(
            artifact.model, data, model_contract=artifact.contract,
            batch_size=config['inference']['batch_size'],
        )
    else:
        result = evaluate_prepared_data(
            artifact, data, expected_role=config['dataset']['expected_role'],
            batch_size=config['inference']['batch_size'],
        )
    metadata['execution_duration_seconds'] = perf_counter() - started
    return save_prepared_run(config, artifact, data, result, metadata=metadata)
