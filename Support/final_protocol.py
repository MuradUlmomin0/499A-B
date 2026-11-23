"""Freeze the existing 48 final-round models before CP7 held-out scoring."""
from Support.Utils.runtime_paths import load_json as runtime_json, load_json_file as runtime_json_file, load_yaml as runtime_yaml, safe_load_yaml as runtime_safe_yaml
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from Support.data.latest_audit import source_hash
from Support.Utils.runtime_paths import packaged, historical_inventory, verify_code, authenticate_recorded, original_member

REPOSITORY = Path(__file__).resolve().parents[1]
METHODS = ('fedavg', 'fedmade', 'temporal', 'fusion')
CONDITIONS = ('iid', 'dirichlet')
SEEDS = (42, 43, 44)


def read(path):
    return runtime_json(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def expected_cases():
    return [dict(name=f'{method}_{"poisoned" if attacked else "clean"}_{condition}_seed{seed}',
                 method=method, condition=condition, seed=seed, attacked=attacked)
            for method in METHODS for condition in CONDITIONS for seed in SEEDS
            for attacked in (False, True)]


def validate_cases(cases):
    expected = expected_cases()
    if len(cases) != 48 or any(any(e.get(k) != v for k, v in c.items()) for e, c in zip(cases, expected)):
        raise ValueError('Exactly the ordered 48 unique CP7 cases are required')


def fingerprints(paths):
    return {str(Path(p).resolve()): source_hash(p) for p in sorted(set(map(str, paths)))}


def authenticate(inventory):
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError('Empty fingerprint inventory')
    verify_code()
    for path, digest in inventory.items():
        if not authenticate_recorded(path, digest):
            raise ValueError('Frozen input changed: ' + path)


def required_sources():
    if packaged(REPOSITORY):
        return set(historical_inventory(7)['source_sha256'])
    paths = [REPOSITORY/'main.py', REPOSITORY/'Requirements.txt', REPOSITORY/'Support/docs/cp7_protocol.md']
    paths += [p for p in (REPOSITORY/'Support').rglob('*.py')
              if not any(x in p.parts for x in ['results','docs','__pycache__'])]
    return {str(p.resolve()) for p in paths}


def required_inputs(entries):
    cp1=read(REPOSITORY/'Support/docs/cp1_evidence/frozen_manifest.json')
    prepared=REPOSITORY/cp1['prepared_directory']
    paths={str((prepared/k).resolve()) for k in cp1['prepared_file_sha256']}
    for cp in range(1,7):
        paths.update(str(p.resolve()) for p in (REPOSITORY/f'Support/docs/cp{cp}_evidence').glob('*.json'))
    for e in entries:
        paths.update(str(p.resolve()) for p in Path(e['run_path']).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    return paths


def build_registry():
    """Select only primary models from the completed checkpoint registries."""
    clean = {m: read(REPOSITORY/f'Support/results/cp{cp}_v1/matrix/comparison.json')['runs']
             for m, cp in [('fedavg', 2), ('fedmade', 3), ('temporal', 4)]}
    poison = read(REPOSITORY/'Support/docs/cp5_evidence/primary_registry.json')
    fusion = read(REPOSITORY/'Support/docs/cp6_evidence/primary_registry.json')
    result = []
    for case in expected_cases():
        pool = fusion if case['method'] == 'fusion' else poison if case['attacked'] else clean[case['method']]
        matches = [e for e in pool if e['condition'] == case['condition'] and e['seed'] == case['seed']
                   and not e.get('control', False) and not e.get('repeat', False)
                   and (case['method'] == 'fusion' and e['attacked'] == case['attacked']
                        or case['method'] != 'fusion' and (not case['attacked'] or e['method'] == case['method']))]
        if len(matches) != 1:
            raise ValueError('Missing/ambiguous primary model: ' + case['name'])
        path = Path(matches[0]['run_path']).resolve()
        v, meta = read(path/'verification.json'), read(path/'metadata.json')
        if v['testing_only'] or v['test_scored'] or not v['reloaded_validation_exact']:
            raise ValueError('Not an authenticated research model: ' + str(path))
        if v['rounds'] != 20 or v['train_rows'] != 500000 or v['validation_rows'] != 752478:
            raise ValueError('Different training/evaluation budget')
        if meta['seed'] != case['seed'] or meta['evaluation_role'] != 'validation':
            raise ValueError('Different model seed/role')
        result.append({**case, 'run_path': str(path), 'checkpoint': 6 if case['method'] == 'fusion' else 5 if case['attacked'] else {'fedavg':2, 'fedmade':3, 'temporal':4}[case['method']]})
    validate_cases(result)
    for condition in CONDITIONS:
        for seed in SEEDS:
            selected = [e for e in result if (e['condition'], e['seed']) == (condition, seed)]
            ref = Path(selected[0]['run_path'])
            rm, rv = read(ref/'metadata.json'), read(ref/'verification.json')
            for e in selected:
                path = Path(e['run_path']); m, v = read(path/'metadata.json'), read(path/'verification.json')
                if m['federated_inputs'] != rm['federated_inputs']:
                    raise ValueError('Unmatched client/validation inputs')
                for key in ['initial_state_sha256', 'rounds', 'train_rows', 'client_rows', 'validation_rows']:
                    if v[key] != rv[key]: raise ValueError('Unmatched training control: ' + key)
            truths = [read(Path(e['run_path'])/'attack_ground_truth.json') for e in selected if e['attacked']]
            if any(t != truths[0] for t in truths):
                raise ValueError('Unmatched attack ground truth')
    return result


def _archive_identity(entries):
    """Authenticate pre-existing model/log bytes against historical archives."""
    checked = []
    for cp in range(2, 7):
        evidence = REPOSITORY/f'Support/docs/cp{cp}_evidence'
        archive = evidence/('run_logs_and_models.zip' if cp == 2 else 'logs_models_and_diagnostics.zip' if cp in [3,4] else 'runs_and_diagnostics.zip')
        completion = read(evidence/'verification.json')
        recorded = completion.get('logs_and_models_sha256', completion.get('diagnostic_archive_sha256', completion.get('runs_archive_sha256')))
        if recorded is not None and source_hash(archive) != recorded:
            raise ValueError('Historical run archive changed')
        with ZipFile(archive) as z:
            for e in entries:
                if e['checkpoint'] != cp: continue
                path = Path(e['run_path'])
                prefix = (f'{e["condition"]}_seed{e["seed"]}' if cp <= 4 else
                          f'{e["method"]}_{e["condition"]}_seed{e["seed"]}' if cp == 5 else str(path.relative_to(REPOSITORY)))
                if packaged(REPOSITORY) and cp == 6:
                    prefix = original_member(path.relative_to(REPOSITORY))
                filenames=['model_artifact.pt', 'metadata.json', 'config.yaml', 'round_metrics.csv', 'aggregation_weights.csv']
                if cp>=4:filenames.append('verification.json')
                for filename in filenames:
                    with z.open(prefix+'/'+filename) as stream:
                        archive_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
                    if source_hash(path/filename) != archive_hash:
                        raise ValueError('Model/log differs from historical archive: ' + str(path/filename))
                checked.append(e['name'])
    if len(checked) != 48: raise ValueError('Incomplete archive authentication')
    return checked


def freeze(output):
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError('Refusing to overwrite a final-test freeze')
    entries = build_registry()
    cp1 = read(REPOSITORY/'Support/docs/cp1_evidence/frozen_manifest.json')
    prepared = REPOSITORY/cp1['prepared_directory']
    original = {str(prepared/k): v for k,v in cp1['prepared_file_sha256'].items()}
    authenticate(original)
    authenticate(read(REPOSITORY/'Support/docs/cp6_evidence/frozen_manifest.json')['code_sha256'])
    authenticated = _archive_identity(entries)
    record = dict(schema_version=1, protocol='cp7_final_comparison_v1', cases=entries,
                  test_manifest=str(prepared/'test.manifest.json'), test_rows=724992,
                  batch_size=128, device='cpu', numerical_threads=1, training_performed=False,
                  model_selection='previously_frozen_final_round', test_scored=False,
                  archive_authenticated_cases=authenticated, source_sha256=fingerprints(required_sources()),
                  input_sha256=fingerprints(required_inputs(entries)))
    output.parent.mkdir(parents=True, exist_ok=True)
    write_new(output, record)
    return record


def load_freeze(path, *, full=True):
    value = read(path)
    if value.get('protocol') != 'cp7_final_comparison_v1' or value.get('schema_version') != 1:
        raise ValueError('Invalid CP7 freeze')
    validate_cases(value['cases'])
    if (value['test_rows'] != 724992 or value['batch_size'] != 128 or value['device'] != 'cpu'
            or value['numerical_threads'] != 1 or value['training_performed'] or value['test_scored']
            or value['model_selection'] != 'previously_frozen_final_round'):
        raise ValueError('Final-evaluation settings changed')
    if set(value['source_sha256']) != required_sources() or set(value['input_sha256']) != required_inputs(value['cases']):
        raise ValueError('Incomplete final-comparison inventory')
    if value['archive_authenticated_cases'] != [c['name'] for cp in range(2,7) for c in value['cases'] if c['checkpoint']==cp]:
        raise ValueError('Incomplete historical archive authentication')
    if build_registry() != value['cases']: raise ValueError('Frozen registry identity changed')
    authenticate(value['source_sha256'])
    if full:
        authenticate(value['input_sha256'])
    return value
