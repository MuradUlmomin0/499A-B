"""Independent random streams and recorded execution environment."""
import os
import random
import platform
import importlib.metadata
import subprocess
from datetime import datetime, timezone
import numpy as np
import torch


def seed_everything(seed, deterministic=True):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False


def environment_metadata(
    seed, device, deterministic, repository, *, testing_only=True, purpose=None,
):
    """Record the execution environment and an explicit run-purpose declaration.

    The original four positional arguments keep Milestone A's defaults:
    testing-only synthetic framework verification. Prepared-run callers may
    supply testing_only and purpose by keyword. A non-testing declaration
    requires an explicit purpose; it does not establish real-data provenance,
    completed training, correct preprocessing, or research validity.

    Args:
        seed: Configured experiment seed; recording it does not set any RNG.
        device: Configured device, stored as a string; availability is not checked.
        deterministic: Configured deterministic setting, recorded without changing
            PyTorch's execution settings.
        repository: Repository path used for read-only Git metadata queries.
        testing_only: Explicit boolean describing the run's declared purpose.
            The prepared runner must combine model/data flags before calling;
            save_prepared_run() also retains any testing-only source declaration.
        purpose: Nonblank description without surrounding whitespace. None uses
            the existing synthetic description only when testing_only is True.

    Returns:
        A dictionary with the seed, package versions, device, deterministic
        setting, UTC timestamp, available Git commit/status, and purpose flags.
        Git fields are None if the Git command is unavailable or fails.

    Raises:
        ValueError: testing_only is not a boolean, or purpose is missing/invalid.
        importlib.metadata.PackageNotFoundError: A required package is absent.

    This function does not seed generators, load artifacts, or write logs.
    Call seed_everything() separately when starting an actual experiment.
    """
    if type(testing_only) is not bool:
        raise ValueError('testing_only must be an explicit boolean')
    if purpose is None:
        if not testing_only:
            raise ValueError('An explicit purpose is required when testing_only is False')
        purpose = 'Synthetic framework verification; not research results'
    elif (not isinstance(purpose, str) or not purpose.strip()
          or purpose != purpose.strip()):
        raise ValueError('purpose must be a nonblank string without surrounding whitespace')

    def git(*args):
        try:
            # Leading spaces distinguish Git's status columns. Trim line
            # endings only so recorded status retains that information.
            return subprocess.check_output(['git', '-c', f'safe.directory={repository}', '-C', str(repository), *args], text=True, stderr=subprocess.DEVNULL).rstrip('\r\n')
        except (OSError, subprocess.CalledProcessError):
            return None
    return {'seed': seed, 'python': platform.python_version(), 'pytorch': torch.__version__,
            'numpy': np.__version__, 'scikit_learn': importlib.metadata.version('scikit-learn'),
            'pyyaml': importlib.metadata.version('PyYAML'), 'device': str(device), 'deterministic': deterministic,
            'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'git_commit': git('rev-parse','HEAD'),
            'git_status': git('status','--porcelain'), 'testing_only': testing_only,
            'purpose': purpose}
