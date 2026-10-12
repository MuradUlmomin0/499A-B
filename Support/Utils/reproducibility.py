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


def environment_metadata(seed, device, deterministic, repository):
    def git(*args):
        try:
            return subprocess.check_output(['git', '-c', f'safe.directory={repository}', '-C', str(repository), *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {'seed': seed, 'python': platform.python_version(), 'pytorch': torch.__version__,
            'numpy': np.__version__, 'scikit_learn': importlib.metadata.version('scikit-learn'),
            'pyyaml': importlib.metadata.version('PyYAML'), 'device': str(device), 'deterministic': deterministic,
            'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'git_commit': git('rev-parse','HEAD'),
            'git_status': git('status','--porcelain'), 'testing_only': True,
            'purpose': 'Synthetic framework verification; not research results'}
