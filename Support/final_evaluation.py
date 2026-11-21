"""Sequential, single-pass CP7 test scoring and authenticated offline checks."""
import argparse
import gc
import json
from pathlib import Path
import resource
import subprocess
import sys
import uuid
from time import perf_counter

import numpy as np
import torch
from threadpoolctl import threadpool_limits, threadpool_info
from torch.utils.data import DataLoader

from Support.data.latest_audit import source_hash
from Support.data.prepared import load_prepared_data
from Support.evaluation.metrics import classification_metrics
from Support.evaluation.predict import _check_logits
from Support.final_protocol import authenticate, fingerprints, load_freeze, read, write_new, freeze
from Support.models.artifact import load_model_artifact
from Support.federated_runner import _state_sha256
from Support.Utils.reproducibility import seed_everything, environment_metadata


def metrics_from_evidence(truth, predictions, losses, class_names):
    """Cross-check shared metrics with direct confusion-matrix equations."""
    truth, predictions, losses = map(np.asarray, (truth, predictions, losses))
    n, c = len(truth), len(class_names)
    if (not n or truth.ndim != 1 or predictions.shape != truth.shape or losses.shape != truth.shape
            or truth.dtype.kind not in 'iu' or predictions.dtype.kind not in 'iu'
            or losses.dtype.kind not in 'f' or not np.isfinite(losses).all() or (losses < 0).any()
            or (truth < 0).any() or (truth >= c).any() or (predictions < 0).any() or (predictions >= c).any()):
        raise ValueError('Invalid saved prediction/loss evidence')
    result = classification_metrics(truth, predictions, c)
    cm = np.bincount(c*truth.astype(np.int64)+predictions.astype(np.int64), minlength=c*c).reshape(c,c)
    support, predicted, tp = cm.sum(axis=1), cm.sum(axis=0), cm.diagonal()
    precision = np.divide(tp, predicted, out=np.zeros(c), where=predicted>0)
    recall = np.divide(tp, support, out=np.zeros(c), where=support>0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(c), where=precision+recall>0)
    manual = dict(accuracy=float(tp.sum()/n), balanced_accuracy=float(recall[support>0].mean()),
                  macro_precision=float(precision.mean()), macro_recall=float(recall.mean()),
                  macro_f1=float(f1.mean()), weighted_f1=float(np.dot(f1,support)/n))
    for k,v in manual.items():
        if not np.isclose(v, result[k], rtol=0, atol=1e-12): raise ValueError('Metric equation mismatch: '+k)
    if cm.tolist() != result['confusion_matrix']: raise ValueError('Confusion-matrix mismatch')
    result['loss'] = float(losses.sum(dtype=np.float64)/n)
    for row in result['per_class']: row['class_name'] = class_names[row['class_id']]
    return result


def score_model(artifact, data, *, batch_size):
    """One forward pass per row, retaining small prediction/loss arrays only."""
    if data.manifest.role != 'test': raise ValueError('Final scoring requires test role')
    if type(batch_size) is not int or batch_size <= 0: raise ValueError('Invalid batch size')
    artifact.contract.assert_compatible(data.manifest.contract)
    n = len(data.dataset)
    predictions, losses = np.empty(n,dtype=np.int64), np.empty(n,dtype=np.float64)
    loader = DataLoader(data.dataset,batch_size=batch_size,shuffle=False,num_workers=0,
                        generator=torch.Generator().manual_seed(0))
    model = artifact.model
    previous = [(m,m.training) for m in model.modules()]
    model.eval(); cursor = 0
    try:
        with torch.inference_mode():
            for x,y in loader:
                if not torch.isfinite(x).all(): raise ValueError('Nonfinite features')
                logits = model(x)
                _check_logits(logits,len(x),artifact.contract.num_classes)
                end = cursor+len(x)
                predictions[cursor:end] = logits.argmax(dim=1).numpy()
                losses[cursor:end] = torch.nn.functional.cross_entropy(logits.double(),y,reduction='none').numpy()
                cursor = end
    finally:
        for module, mode in previous: module.training = mode
    if cursor != n: raise ValueError('Incomplete test inference')
    return predictions, losses


def verify_case(directory, case, frozen, freeze_path):
    """Verify saved predictions/metrics; never forward the model again."""
    directory = Path(directory)
    record = read(directory/'result.json')
    if record.get('evaluation_role')!='test' or record.get('test_scored') is not True or record.get('training_performed') is not False:
        raise ValueError('Result evaluation/training declarations differ')
    if record['case'] != case or record['freeze_sha256'] != source_hash(freeze_path):
        raise ValueError('Result belongs to another model/freeze')
    expected = {str((directory/p).resolve()) for p in ['predictions.npz','result.json']}
    inventory = read(directory/'artifacts.json')
    if set(inventory) != expected: raise ValueError('Incomplete result artifact inventory')
    authenticate(inventory)
    manifest = Path(frozen['test_manifest'])
    truth = np.load(manifest.parent/'test_labels.npy',mmap_mode='r',allow_pickle=False)
    with np.load(directory/'predictions.npz',allow_pickle=False) as z:
        if set(z.files) != {'predictions','losses'}: raise ValueError('Invalid evidence arrays')
        metrics = metrics_from_evidence(truth,z['predictions'],z['losses'],read(manifest)['contract']['class_names'])
    if metrics != record['metrics'] or record['test_rows'] != frozen['test_rows']:
        raise ValueError('Saved test metrics differ from predictions')
    if record['model_sha256'] != source_hash(Path(case['run_path'])/'model_artifact.pt'):
        raise ValueError('Result model changed')
    return record


def _evaluate_one(freeze_path, name, directory):
    started = perf_counter()
    frozen = load_freeze(freeze_path,full=False)
    matches = [c for c in frozen['cases'] if c['name'] == name]
    if len(matches) != 1: raise ValueError('Unknown final comparison case')
    case = matches[0]; path = Path(case['run_path']); manifest = Path(frozen['test_manifest'])
    required = [str(path/'model_artifact.pt'),str(path/'verification.json'),str(manifest),
                str(manifest.parent/'test.npz'),str(manifest.parent/'test_labels.npy'),str(manifest.parent/'preprocessing.json')]
    authenticate({p:frozen['input_sha256'][p] for p in required})
    directory = Path(directory); directory.mkdir(parents=True,exist_ok=False)
    try:
        with threadpool_limits(limits=1):
            torch.set_num_threads(1); seed_everything(case['seed'],True)
            artifact = load_model_artifact(path/'model_artifact.pt')
            if _state_sha256(artifact.model.state_dict()) != read(path/'verification.json')['final_state_sha256']:
                raise ValueError('Exported state differs from verified final round')
            data = load_prepared_data(manifest,expected_role='test',expected_contract=artifact.contract)
            if artifact.testing_only or data.manifest.testing_only or not artifact.training_completed:
                raise ValueError('Final research scoring requires trained real-data artifacts')
            if len(data.dataset) != frozen['test_rows']: raise ValueError('Different final-test population')
            before = {k:v.clone() for k,v in artifact.model.state_dict().items()}
            pred, losses = score_model(artifact,data,batch_size=frozen['batch_size'])
            if any(not torch.equal(v,before[k]) for k,v in artifact.model.state_dict().items()):
                raise ValueError('Inference changed model weights')
            truth = data.dataset.tensors[1].numpy()
            if not np.array_equal(truth,np.load(manifest.parent/'test_labels.npy',allow_pickle=False)):
                raise ValueError('NPZ truth differs from frozen aligned labels')
            metrics = metrics_from_evidence(truth,pred,losses,artifact.contract.class_names)
            np.savez_compressed(directory/'predictions.npz',predictions=pred,losses=losses)
            pools = threadpool_info()
            if any(p['num_threads'] != 1 for p in pools): raise ValueError('Numerical thread limit violated')
            result = dict(case=case, freeze_sha256=source_hash(freeze_path),model_sha256=artifact.artifact_sha256,
                          evaluation_role='test',test_scored=True,training_performed=False,test_rows=len(truth),
                          metrics=metrics,numerical_threadpools=pools,pytorch_threads=torch.get_num_threads(),
                          inference_and_metrics_seconds=perf_counter()-started,
                          peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            result['environment']=environment_metadata(case['seed'],'cpu',True,Path(__file__).resolve().parents[1],
                testing_only=False,purpose='CP7 frozen held-out evaluation without training')
            if result['peak_process_rss_kib'] > int(3.5*1024**2): raise MemoryError('Final scoring exceeded 3.5 GiB')
            write_new(directory/'result.json',result)
            write_new(directory/'artifacts.json',fingerprints([directory/'predictions.npz',directory/'result.json']))
            verify_case(directory,case,frozen,freeze_path)
    except BaseException as exc:
        write_new(directory/'failure.json',dict(error=type(exc).__name__,message=str(exc),case=case))
        raise
    return result


def _save_failure(output, exc, operation):
    output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    write_new(output.parent/(output.name+'_failure_'+uuid.uuid4().hex+'.json'),
              dict(operation=operation,error=type(exc).__name__,message=str(exc),output=str(output)))


def evaluate_one(freeze_path, name, directory):
    try:
        return _evaluate_one(freeze_path,name,directory)
    except BaseException as exc:
        _save_failure(directory,exc,'evaluate_one')
        raise


def preflight(freeze_path, output):
    """Measure input loading and thread limits without any predictions."""
    frozen=load_freeze(freeze_path)
    with threadpool_limits(limits=1):
        torch.set_num_threads(1)
        artifact=load_model_artifact(Path(frozen['cases'][0]['run_path'])/'model_artifact.pt')
        data=load_prepared_data(frozen['test_manifest'],expected_role='test',expected_contract=artifact.contract)
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if len(data.dataset)!=frozen['test_rows'] or rss>int(3.5*1024**2):raise ValueError('Final-data preflight failed')
        pools=threadpool_info()
        if any(p['num_threads']!=1 for p in pools):raise ValueError('Numerical threads exceed one')
        record=dict(test_rows=len(data.dataset),peak_process_rss_kib=rss,numerical_threadpools=pools,
                    model_forward_called=False,test_scored=False,training_performed=False,freeze_sha256=source_hash(freeze_path))
        record['environment']=environment_metadata(42,'cpu',True,Path(__file__).resolve().parents[1],
            testing_only=False,purpose='CP7 input-only preflight')
        write_new(output,record)
    return record


def find_completed_attempt(output, case, frozen, freeze_path):
    """Recover a verified worker result if the parent exited before indexing it."""
    complete=[]
    for directory in sorted(Path(output).glob(case['name']+'_attempt*')):
        if not directory.is_dir(): continue
        if (directory/'result.json').exists() or (directory/'artifacts.json').exists():
            # A claimed complete result must verify; corruption is never retried silently.
            verify_case(directory,case,frozen,freeze_path)
            complete.append(directory)
    if len(complete)>1: raise ValueError('Multiple complete unindexed attempts')
    return complete[0] if complete else None


def _run_matrix(freeze_path, output, *, resume=False):
    frozen = load_freeze(freeze_path)
    measured=read(Path(freeze_path).parent/'preflight.json')
    if (measured['freeze_sha256']!=source_hash(freeze_path) or measured['test_scored']
            or measured['model_forward_called'] or measured['test_rows']!=frozen['test_rows']
            or measured['peak_process_rss_kib']>int(3.5*1024**2)
            or any(p['num_threads']!=1 for p in measured['numerical_threadpools'])):
        raise ValueError('Missing/stale final-data preflight')
    output = Path(output).resolve(); output.mkdir(parents=True,exist_ok=resume)
    index = output/'progress.json'
    entries = read(index) if index.exists() else []
    cases = frozen['cases']
    if len(entries)>48 or [e['name'] for e in entries] != [c['name'] for c in cases[:len(entries)]]:
        raise ValueError('Invalid final-test matrix prefix')
    for i,case in enumerate(cases):
        if i < len(entries):
            entry=entries[i]; directory=Path(entry['directory'])
            if entry['artifacts_sha256'] != source_hash(directory/'artifacts.json'): raise ValueError('Cached evidence changed')
            verify_case(directory,case,frozen,freeze_path)
            print('Retained '+case['name'],flush=True);continue
        directory=find_completed_attempt(output,case,frozen,freeze_path)
        if directory is None:
            attempt=1
            while (output/f'{case["name"]}_attempt{attempt}').exists() or (output/f'{case["name"]}_attempt{attempt}.log').exists():attempt+=1
            directory=output/f'{case["name"]}_attempt{attempt}'
            print(f'Scoring {i+1}/48 '+case['name'],flush=True)
            with (output/f'{case["name"]}_attempt{attempt}.log').open('x') as log:
                subprocess.run([sys.executable,'-m','Support.final_evaluation','one','--freeze',str(Path(freeze_path).resolve()),
                                '--name',case['name'],'--output',str(directory)],stdout=log,stderr=subprocess.STDOUT,check=True)
        else:
            print('Recovered complete unindexed '+case['name'],flush=True)
        verify_case(directory,case,frozen,freeze_path)
        entries.append(dict(name=case['name'],directory=str(directory),artifacts_sha256=source_hash(directory/'artifacts.json')))
        temporary=output/'progress.tmp';temporary.write_text(json.dumps(entries,indent=2)+'\n');temporary.replace(index)
        print('Verified '+case['name'],flush=True)
    load_freeze(freeze_path)
    return entries


def run_matrix(freeze_path, output, *, resume=False):
    try:
        return _run_matrix(freeze_path,output,resume=resume)
    except BaseException as exc:
        _save_failure(output,exc,'matrix')
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
    f=s.add_parser('freeze');f.add_argument('--output',type=Path,required=True)
    for action in ['matrix','one','verify','preflight']:
        a=s.add_parser(action);a.add_argument('--freeze',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
        if action=='matrix':a.add_argument('--resume',action='store_true')
        if action=='one':a.add_argument('--name',required=True)
    a=p.parse_args()
    try:
        if a.action=='freeze':freeze(a.output);print('Final comparison frozen before scoring.')
        elif a.action=='matrix':run_matrix(a.freeze,a.output,resume=a.resume)
        elif a.action=='one':evaluate_one(a.freeze,a.name,a.output)
        elif a.action=='preflight':preflight(a.freeze,a.output);print('Input-only memory preflight passed.')
        else:
            frozen=load_freeze(a.freeze);entries=read(a.output/'progress.json')
            if len(entries)!=48:raise ValueError('Final matrix is incomplete')
            for e,c in zip(entries,frozen['cases']):
                if e['name']!=c['name'] or e['artifacts_sha256']!=source_hash(Path(e['directory'])/'artifacts.json'):raise ValueError('Matrix identity changed')
                verify_case(e['directory'],c,frozen,a.freeze)
            print('All 48 saved test results verified offline.')
    except BaseException as exc:
        _save_failure(a.output,exc,a.action)
        raise


if __name__=='__main__':main()
