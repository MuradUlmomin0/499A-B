"""Prepared configuration checks using temporary YAML files only.

No model or dataset is created, loaded, or trained. CUDA is tested only as a
configuration value; device availability belongs to the future workflow runner.
"""

from copy import deepcopy

import pytest
import yaml

from support.utils.prepared_config import load_prepared_config, validate_prepared_config


@pytest.fixture
def settings():
    """Supply independent, valid prepared-run settings to each test case."""
    return {
        'schema_version': 1,
        'mode': 'evaluate',
        'experiment': {
            'name': 'prepared_test',
            'seed': 42,
            'device': 'cpu',
            'deterministic': True,
            'output_dir': '../results',
        },
        'model': {'artifact_path': '../models/model.pt'},
        'dataset': {'manifest_path': '../data/manifest.json', 'expected_role': 'test'},
        'inference': {'batch_size': 256},
    }


def write_config(tmp_path, settings):
    """Write a temporary YAML document, including a folder name with spaces."""
    folder = tmp_path / 'configuration folder'
    folder.mkdir(exist_ok=True)
    path = folder / 'prepared.yaml'
    path.write_text(yaml.safe_dump(settings, sort_keys=False), encoding='utf-8')
    return path


def set_value(settings, location, value):
    """Change one root or section setting for a parameterized invalid-input case."""
    if '.' in location:
        section, field = location.split('.')
        settings[section][field] = value
    else:
        settings[location] = value


@pytest.mark.parametrize('mode,device', [('predict', 'cpu'), ('evaluate', 'cuda')])
def test_loads_settings_without_opening_artifacts_or_creating_outputs(tmp_path, settings, mode, device):
    settings['mode'] = mode
    settings['experiment']['device'] = device
    path = write_config(tmp_path, settings)
    content = path.read_bytes()

    loaded = load_prepared_config(path)

    assert loaded['mode'] == mode
    assert loaded['experiment']['device'] == device
    for section, field in (
        ('experiment', 'output_dir'), ('model', 'artifact_path'), ('dataset', 'manifest_path'),
    ):
        expected = (path.parent / settings[section][field]).resolve()
        assert loaded[section][field] == str(expected)
        assert not expected.exists()
    assert path.read_bytes() == content


@pytest.mark.parametrize('seed', [0, 2**32 - 1])
def test_accepts_seed_boundaries(settings, seed):
    settings['experiment']['seed'] = seed

    assert validate_prepared_config(settings)['experiment']['seed'] == seed


@pytest.mark.parametrize('role', ['train', 'validation', 'auxiliary', 'test', 'predict'])
def test_all_explicit_roles_can_be_requested(settings, role):
    settings['dataset']['expected_role'] = role

    assert validate_prepared_config(settings)['dataset']['expected_role'] == role


def test_validation_returns_independent_sections_without_resolving_paths(settings):
    before = deepcopy(settings)

    validated = validate_prepared_config(settings)

    assert validated == settings
    assert validated is not settings
    for section in ('experiment', 'model', 'dataset', 'inference'):
        assert validated[section] is not settings[section]
    validated['experiment']['seed'] = 1
    validated['model']['artifact_path'] = 'other.pt'
    validated['dataset']['expected_role'] = 'validation'
    validated['inference']['batch_size'] = 3
    assert settings == before


@pytest.mark.parametrize('missing', [
    'schema_version', 'mode', 'experiment', 'model', 'dataset', 'inference',
])
def test_rejects_missing_root_fields(settings, missing):
    del settings[missing]

    with pytest.raises(ValueError, match='fields'):
        validate_prepared_config(settings)


@pytest.mark.parametrize('extra', ['training', 'testing_only'])
def test_rejects_unsupported_training_and_purpose_overrides(settings, extra):
    settings[extra] = False

    with pytest.raises(ValueError, match='fields'):
        validate_prepared_config(settings)


@pytest.mark.parametrize('section', ['experiment', 'model', 'dataset', 'inference'])
def test_rejects_missing_section_settings(settings, section):
    del settings[section][next(iter(settings[section]))]

    with pytest.raises(ValueError, match=section):
        validate_prepared_config(settings)


@pytest.mark.parametrize('section,field', [
    ('model', 'input_dim'), ('model', 'testing_only'), ('dataset', 'testing_only'),
])
def test_artifact_metadata_cannot_be_overridden_from_configuration(settings, section, field):
    settings[section][field] = False

    with pytest.raises(ValueError, match=section):
        validate_prepared_config(settings)


@pytest.mark.parametrize('section', ['experiment', 'model', 'dataset', 'inference'])
def test_sections_must_be_mappings(settings, section):
    settings[section] = []

    with pytest.raises(ValueError, match=section):
        validate_prepared_config(settings)


@pytest.mark.parametrize('location,value', [
    ('schema_version', True), ('schema_version', 2), ('mode', 'train'), ('mode', None),
    ('experiment.seed', True), ('experiment.seed', -1), ('experiment.seed', 2**32),
    ('experiment.name', '../run'), ('experiment.name', ' run'), ('experiment.name', ''),
    ('experiment.device', 'auto'), ('experiment.deterministic', 'true'),
    ('inference.batch_size', True), ('inference.batch_size', 0), ('inference.batch_size', 1.5),
    ('dataset.expected_role', 'evaluation'),
])
def test_rejects_invalid_scalar_settings(settings, location, value):
    set_value(settings, location, value)

    with pytest.raises(ValueError):
        validate_prepared_config(settings)


@pytest.mark.parametrize('location,value', [
    ('experiment.output_dir', ''), ('experiment.output_dir', 'bad\x00path'),
    ('model.artifact_path', '  '), ('model.artifact_path', None),
    ('dataset.manifest_path', ' manifest.json'), ('dataset.manifest_path', 'manifest.json '),
])
def test_rejects_invalid_paths(settings, location, value):
    set_value(settings, location, value)

    with pytest.raises(ValueError, match='path'):
        validate_prepared_config(settings)


def test_relative_paths_are_independent_of_working_directory(tmp_path, settings, monkeypatch):
    path = write_config(tmp_path, settings)
    expected = load_prepared_config(path)
    unrelated = tmp_path / 'unrelated'
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)

    assert load_prepared_config(path) == expected


def test_accepts_a_relative_config_filename(tmp_path, settings, monkeypatch):
    path = write_config(tmp_path, settings)
    monkeypatch.chdir(tmp_path)

    loaded = load_prepared_config(path.relative_to(tmp_path))

    assert loaded['model']['artifact_path'] == str((tmp_path / 'models/model.pt').resolve())


def test_absolute_paths_are_preserved(tmp_path, settings):
    locations = (
        ('experiment', 'output_dir'), ('model', 'artifact_path'), ('dataset', 'manifest_path'),
    )
    for index, (section, field) in enumerate(locations):
        settings[section][field] = str(tmp_path / f'absolute target {index}')
    path = write_config(tmp_path, settings)

    loaded = load_prepared_config(path)

    assert all(loaded[section][field] == settings[section][field] for section, field in locations)


@pytest.mark.parametrize('literal', ['~/model.pt', '${PRIVATE_SHIELD_CONFIG_TEST_DIR}/model.pt'])
def test_paths_are_literal_without_user_or_environment_expansion(tmp_path, settings, monkeypatch, literal):
    monkeypatch.setenv('PRIVATE_SHIELD_CONFIG_TEST_DIR', str(tmp_path / 'expanded'))
    settings['model']['artifact_path'] = literal
    path = write_config(tmp_path, settings)

    assert load_prepared_config(path)['model']['artifact_path'] == str((path.parent / literal).resolve())


def test_normalized_settings_can_be_saved_and_reloaded(tmp_path, settings):
    path = write_config(tmp_path, settings)
    normalized = load_prepared_config(path)
    log_copy = tmp_path / 'logged.yaml'
    log_copy.write_text(yaml.safe_dump(normalized), encoding='utf-8')

    assert load_prepared_config(log_copy) == normalized
    assert validate_prepared_config(normalized) == normalized


@pytest.mark.parametrize('setting', ['mode', 'seed', 'artifact_path', 'manifest_path', 'batch_size'])
def test_duplicate_keys_are_rejected_at_every_schema_level(tmp_path, settings, setting):
    text = yaml.safe_dump(settings, sort_keys=False)
    lines = text.splitlines(keepends=True)
    target = next(index for index, line in enumerate(lines) if line.lstrip().startswith(setting + ':'))
    lines.insert(target, lines[target])
    path = tmp_path / 'duplicate.yaml'
    path.write_text(''.join(lines), encoding='utf-8')

    # Even identical duplicate values must be rejected: a later edit must not
    # silently change which declaration is used for a reproducible experiment.
    with pytest.raises(ValueError, match='Duplicate'):
        load_prepared_config(path)


def test_yaml_merge_keys_are_rejected(tmp_path, settings):
    text = yaml.safe_dump(settings, sort_keys=False)
    text = text.replace('inference:\n  batch_size: 256', 'inference:\n  <<: {batch_size: 256}')
    path = tmp_path / 'merge.yaml'
    path.write_text(text, encoding='utf-8')

    with pytest.raises(ValueError, match='merge'):
        load_prepared_config(path)


@pytest.mark.parametrize('text', ['1: invalid\n', '? [a, b]\n: invalid\n'])
def test_nonstring_mapping_keys_are_rejected(tmp_path, text):
    path = tmp_path / 'keys.yaml'
    path.write_text(text, encoding='utf-8')

    with pytest.raises(ValueError, match='keys must be strings'):
        load_prepared_config(path)


@pytest.mark.parametrize('text', [
    'experiment: [unterminated',
    'a: 1\n---\na: 2\n',
    '!!python/object/apply:builtins.str [123]',
])
def test_malformed_multiple_document_and_unsafe_yaml_are_rejected(tmp_path, text):
    path = tmp_path / 'bad.yaml'
    path.write_text(text, encoding='utf-8')

    with pytest.raises(ValueError, match='YAML'):
        load_prepared_config(path)


@pytest.mark.parametrize('text', ['!!map [a, b]', '!!map scalar'])
def test_mapping_tags_require_mapping_nodes(tmp_path, text):
    path = tmp_path / 'bad_mapping.yaml'
    path.write_text(text, encoding='utf-8')

    with pytest.raises(ValueError, match='key/value'):
        load_prepared_config(path)


@pytest.mark.parametrize('text', ['', '[]', 'null', 'a scalar'])
def test_empty_and_nonmapping_yaml_documents_are_rejected(tmp_path, text):
    path = tmp_path / 'empty.yaml'
    path.write_text(text, encoding='utf-8')

    with pytest.raises(ValueError, match='fields'):
        load_prepared_config(path)


@pytest.mark.parametrize('kind', ['missing', 'directory'])
def test_configuration_must_be_an_existing_file(tmp_path, kind):
    path = tmp_path / 'config.yaml'
    if kind == 'directory':
        path.mkdir()

    with pytest.raises(FileNotFoundError):
        load_prepared_config(path)


def test_custom_loader_does_not_change_global_yaml_safe_loader():
    # Milestone A uses the ordinary safe loader. The prepared loader's stricter
    # duplicate-key policy must be scoped to its own subclass.
    assert yaml.safe_load('key: first\nkey: second\n') == {'key': 'second'}
