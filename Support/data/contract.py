"""Explicit model/data descriptions; no data loading or preprocessing.

Matching tensor shapes is not sufficient: swapped feature columns or class IDs
can silently change predictions. A model and the prepared data used with it
must declare compatible meanings as well as compatible dimensions.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ModelDataContract:
    """Immutable description supplied alongside a model or prepared dataset.

    ``feature_names`` lists columns in input order. ``class_names[i]`` names
    integer label/output index ``i``. Names and dimensions are supplied by the
    artifact producer; this class never infers a real dataset's taxonomy.

    ``preprocessing_sha256`` identifies the exact preprocessing artifact,
    including its pipeline and fitted parameters. Supply None explicitly for
    no preprocessing. A name such as 'minmax' alone cannot identify which
    fitted transformation was used. Fingerprinting/fitting is outside this API.

    Only the currently implemented MLP architecture is supported. This contract
    describes compatibility, not provenance, split isolation, or research validity.
    """

    model_type: str
    input_dim: int
    num_classes: int
    hidden_dims: tuple[int, ...]
    feature_names: tuple[str, ...]
    class_names: tuple[str, ...]
    preprocessing_sha256: str | None

    def __post_init__(self):
        if self.model_type != 'mlp':
            raise ValueError('Only the implemented mlp model type is supported')
        for name, minimum in (('input_dim', 1), ('num_classes', 2)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')

        # Copy caller-owned lists into tuples so later edits cannot change the
        # recorded feature order, class meaning, or architecture.
        for name in ('hidden_dims', 'feature_names', 'class_names'):
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)) or not values:
                raise ValueError(f'{name} must be a nonempty list or tuple')
            object.__setattr__(self, name, tuple(values))

        if any(type(width) is not int or width < 1 for width in self.hidden_dims):
            raise ValueError('Hidden layer widths must be positive integers')
        for name, count in (('feature_names', self.input_dim),
                            ('class_names', self.num_classes)):
            names = getattr(self, name)
            if len(names) != count:
                raise ValueError(f'{name} length must equal its configured dimension')
            if any(not isinstance(value, str) or not value.strip()
                   or value != value.strip() for value in names):
                raise ValueError(f'{name} must contain nonblank names without surrounding whitespace')
            if len(set(names)) != len(names):
                raise ValueError(f'{name} must contain unique names')

        fingerprint = self.preprocessing_sha256
        if fingerprint is not None and (
            not isinstance(fingerprint, str)
            or re.fullmatch(r'[0-9a-f]{64}', fingerprint) is None
        ):
            raise ValueError('preprocessing_sha256 must be 64 lowercase hexadecimal characters or None')

    def model_config(self) -> dict:
        """Return a fresh dictionary accepted by models.factory.create_model()."""
        return {
            'type': self.model_type,
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'hidden_dims': list(self.hidden_dims),
        }

    def to_dict(self) -> dict:
        """Return a fresh JSON/YAML-compatible description with schema version 1."""
        return {
            'schema_version': 1,
            'model_type': self.model_type,
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'hidden_dims': list(self.hidden_dims),
            'feature_names': list(self.feature_names),
            'class_names': list(self.class_names),
            'preprocessing_sha256': self.preprocessing_sha256,
        }

    @classmethod
    def from_dict(cls, description: Mapping) -> 'ModelDataContract':
        """Validate an explicit description; do not fill missing artifact details."""
        required = {
            'schema_version', 'model_type', 'input_dim', 'num_classes',
            'hidden_dims', 'feature_names', 'class_names', 'preprocessing_sha256',
        }
        if not isinstance(description, Mapping) or set(description) != required:
            raise ValueError('Contract fields must exactly match the version 1 schema')
        if type(description['schema_version']) is not int or description['schema_version'] != 1:
            raise ValueError('Only contract schema version 1 is supported')
        return cls(**{name: description[name] for name in required - {'schema_version'}})

    def assert_compatible(self, other: 'ModelDataContract') -> None:
        """Reject differences between an intended model/data contract pair.

        This compares declarations only. Future loaders must also verify actual
        arrays, label ranges, checkpoint tensors, and preprocessing fingerprints.
        It does not impose a preprocessing/partition policy across all FL clients.
        """
        if not isinstance(other, ModelDataContract):
            raise TypeError('other must be a ModelDataContract')
        fields = (
            'model_type', 'input_dim', 'num_classes', 'hidden_dims',
            'feature_names', 'class_names', 'preprocessing_sha256',
        )
        differences = [name for name in fields if getattr(self, name) != getattr(other, name)]
        if differences:
            raise ValueError('Incompatible model/data contracts: ' + ', '.join(differences))
