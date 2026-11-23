"""Small, CPU-only presentation adapters; no training or holdout scoring."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from threading import RLock
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch

from Support.Utils.prepared_config import load_prepared_config
from Support.Utils.runtime_paths import load_json
from Support.data.manifest import load_data_manifest
from Support.data.prepared import load_prepared_data
from Support.evaluation.predict import predict_prepared_data
from Support.models.artifact import load_model_artifact

ROOT=Path(__file__).resolve().parents[2]
MODELS={
    'fedavg':('FedAvg','cp2_iid_seed42'),
    'fedmade':('FedMADE (adapted)','cp3_iid_seed42'),
    'temporal':('Temporal trust','cp4_iid_seed42'),
    'fusion':('Fusion','cp6_fused_clean_iid_seed42'),
}


class FrontendError(ValueError):
    """A local demonstration input is invalid or incompatible."""


class UnavailableError(FrontendError):
    """An optional asset is absent from this package."""


@dataclass(frozen=True)
class ModelChoice:
    key:str
    label:str
    artifact_path:Path|None
    available:bool
    reason:str=''


@dataclass(frozen=True)
class PredictionView:
    model_key:str
    model_label:str
    model_name:str
    artifact_sha256:str
    frame:pd.DataFrame
    feature_preview:pd.DataFrame
    class_names:tuple[str,...]
    testing_only:bool

    def csv_bytes(self):
        return self.frame.to_csv(index=False).encode('utf-8')


def _local(path,root):
    path=Path(path).resolve()
    if not path.is_relative_to(root):
        raise FrontendError('Demo assets must be inside this project folder.')
    return path


def _checksum(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def _check_asset(path,root):
    """Verify installed assets when the full package inventory is present."""
    path=_local(path,root)
    if not path.is_file():raise UnavailableError('A required local demo asset is missing.')
    actual=_checksum(path)
    inventory=root/'Support/runtime_manifest.json'
    if inventory.is_file():
        try:
            record=json.loads(inventory.read_text());expected=record['assets'][str(path.relative_to(root))]
        except (KeyError,ValueError) as exc:
            raise FrontendError('The demo asset is not listed in the runtime package.') from exc
        if actual!=expected:raise FrontendError('The demo asset does not match its packaged checksum.')
    return actual


class PredictionService:
    """Cache one artifact and serialize inference across browser sessions."""
    def __init__(self,root=ROOT):
        self.root=Path(root).resolve();self._lock=RLock();self._artifact=None;self._cache_key=None
        torch.set_num_threads(1)

    def catalog(self):
        choices=[]
        for key,(label,_) in MODELS.items():
            config=self.root/f'Support/configs/demo_{key}.yaml'
            if key=='fedavg' and not config.is_file():config=self.root/'Support/configs/demo_predict.yaml'
            try:
                c=load_prepared_config(config);path=_local(c['model']['artifact_path'],self.root)
                available=path.is_file()
                choices.append(ModelChoice(key,label,path,available,'' if available else 'Saved model is not included in this package.'))
            except (OSError,ValueError,TypeError):
                choices.append(ModelChoice(key,label,None,False,'Demo configuration is missing or invalid.'))
        return choices

    def _demo(self,contract=None):
        try:
            c=load_prepared_config(self.root/'Support/configs/demo_predict.yaml')
            if c['mode']!='predict' or c['dataset']['expected_role']!='predict':
                raise FrontendError('This screen accepts the bundled prediction demo only.')
            path=_local(c['dataset']['manifest_path'],self.root)
            if path!=self.root/'data/processed/demo_predict/manifest.json':
                raise FrontendError('This screen accepts the bundled prediction demo only.')
            _check_asset(path,self.root)
            m=load_data_manifest(path)
            if m.role!='predict':raise FrontendError('Demo data must have the prediction role.')
            _check_asset(m.data_path,self.root)
            if m.preprocessing_path is not None:_check_asset(m.preprocessing_path,self.root)
            # Check uncompressed sizes and row count before NumPy loads arrays.
            with ZipFile(m.data_path) as archive:
                if sum(i.file_size for i in archive.infolist())>2*1024**2:
                    raise FrontendError('The bundled prediction data exceeds the small-demo limit.')
                with archive.open('features.npy') as stream:
                    version=np.lib.format.read_magic(stream)
                    if version==(1,0):shape,_,_=np.lib.format.read_array_header_1_0(stream)
                    elif version==(2,0):shape,_,_=np.lib.format.read_array_header_2_0(stream)
                    else:raise FrontendError('Unsupported prediction array format.')
                if len(shape)!=2 or not 1<=shape[0]<=20:
                    raise FrontendError('The first dashboard version supports at most 20 bundled records.')
            return load_prepared_data(path,expected_role='predict',expected_contract=contract)
        except FrontendError:raise
        except Exception as exc:
            raise FrontendError('Prepared demo data is missing, invalid, or incompatible with the model contract.') from exc

    def demo_summary(self):
        data=self._demo();c=data.manifest.contract
        return dict(records=len(data.dataset),features=c.input_dim,classes=c.num_classes,
                    hidden_dims=list(c.hidden_dims),testing_only=data.manifest.testing_only)

    def predict(self,model_key):
        with self._lock:
            choice=next((c for c in self.catalog() if c.key==model_key),None)
            if choice is None or not choice.available:
                raise UnavailableError('The selected saved model is unavailable in this package.')
            try:
                if choice.artifact_path.stat().st_size>8*1024**2:
                    raise FrontendError('The selected artifact exceeds the small-demo model limit.')
                digest=_check_asset(choice.artifact_path,self.root);key=(model_key,str(choice.artifact_path),digest)
                if key!=self._cache_key:
                    self._artifact=None;self._cache_key=None
                    artifact=load_model_artifact(choice.artifact_path)
                    if artifact.artifact_sha256!=digest:
                        raise FrontendError('The saved model changed while loading.')
                    if artifact.model_name!=MODELS[model_key][1]:
                        raise FrontendError('The saved model identity does not match the selected experiment.')
                    self._artifact=artifact;self._cache_key=key
                artifact=self._artifact;data=self._demo(artifact.contract)
                result=predict_prepared_data(artifact.model,data,model_contract=artifact.contract,batch_size=128)
                scores=result.probabilities.numpy();ids=result.class_ids.numpy()
                frame=pd.DataFrame(dict(row_index=np.arange(len(ids)),class_id=ids,class_name=result.predicted_labels,
                    model_score=scores[np.arange(len(ids)),ids],model_method=model_key))
                for i in range(scores.shape[1]):frame[f'probability_{i}']=scores[:,i]
                preview=pd.DataFrame(data.dataset.tensors[0].numpy().copy(),columns=artifact.contract.feature_names)
                return PredictionView(model_key,choice.label,artifact.model_name,artifact.artifact_sha256,frame,preview,
                    artifact.contract.class_names,artifact.testing_only or data.manifest.testing_only)
            except FrontendError:raise
            except Exception as exc:
                self._artifact=None;self._cache_key=None
                raise FrontendError('The saved model or prepared inputs could not be loaded. Check their contracts and package files.') from exc


def _number(value,minimum=0,maximum=1):
    if type(value) not in (int,float) or not math.isfinite(value) or not minimum<=value<=maximum:
        raise FrontendError('Saved results contain an invalid metric.')
    return float(value)


def load_research_results(root=ROOT,condition='iid'):
    root=Path(root).resolve();path=root/'Support/results/cp7_v1/analysis/comparison.json'
    if not path.is_file():raise UnavailableError('Saved research results are not included in this package.')
    try:
        _check_asset(path,root);record=load_json(path.read_text())
        if (record['evaluation_role']!='test' or record['test_scored'] is not True
            or record['training_performed'] is not False or record['case_count']!=48):
            raise FrontendError('Expected the saved 48-case final-test comparison.')
        rows=record['summary'];expected={(k,c) for k in MODELS for c in ['iid','dirichlet']}
        if len(rows)!=8 or {(r['method'],r['condition']) for r in rows}!=expected or any(r['n_seeds']!=3 for r in rows):
            raise FrontendError('Expected all four methods, both partitions, and three seeds.')
        if condition not in ('iid','dirichlet'):raise FrontendError('Unknown client partition.')
        result=[]
        for method,(label,_) in MODELS.items():
            row=next(r for r in rows if (r['method'],r['condition'])==(method,condition))
            result.append({'method':method,'Model':label,
                'Clean accuracy (%)':100*_number(row['clean_accuracy_mean']),
                'Poisoned accuracy (%)':100*_number(row['attacked_accuracy_mean']),
                'Clean macro F1':_number(row['clean_macro_f1_mean']),
                'Poisoned macro F1':_number(row['attacked_macro_f1_mean']),
                'Target recall drop (pp)':100*_number(row['recall_degradation_mean'],-1,1)})
        return pd.DataFrame(result)
    except FrontendError:raise
    except Exception as exc:raise FrontendError('Saved research results could not be read or validated.') from exc


def load_training_history(model_key,root=ROOT):
    api=PredictionService(root);choice=next((c for c in api.catalog() if c.key==model_key),None)
    if choice is None or not choice.available:raise UnavailableError('The selected model is unavailable.')
    path=choice.artifact_path.parent/'round_metrics.csv'
    if not path.is_file():raise UnavailableError('Saved training history is not included in this package.')
    try:
        _check_asset(path,api.root);frame=pd.read_csv(path)
        if len(frame)!=20 or frame['round_id'].tolist()!=list(range(1,21)) or not frame['evaluation_role'].eq('validation').all():
            raise FrontendError('Expected 20 ordered rounds of saved validation history.')
        for column in ['accuracy','macro_f1']:
            for value in frame[column]:_number(value)
        return frame[['round_id','accuracy','macro_f1']].copy()
    except FrontendError:raise
    except Exception as exc:raise FrontendError('Saved training history could not be read or validated.') from exc
