"""Independent CP1 artifact, source-row alignment and role/group checks."""
from Support.Utils.runtime_paths import load_json as runtime_json, load_json_file as runtime_json_file, load_yaml as runtime_yaml, safe_load_yaml as runtime_safe_yaml
import gc
import json
from pathlib import Path
import resource
from time import perf_counter

import numpy as np

from Support.data.latest_audit import input_digest, source_hash
from Support.data.prepared import load_prepared_data
from Support.data.split_checks import load_split_evidence, check_disjoint_splits


def verify_prepared(source_dir, prepared_dir):
    """Check all exports without predicting, fitting, or selecting final scores."""
    started=perf_counter();source=Path(source_dir);root=Path(prepared_dir)
    report=runtime_json((root/'preparation_report.json').read_text())
    selection=runtime_json((source/'selection.json').read_text())
    for filename,digest in selection['files'].items():
        if source_hash(source/filename)!=digest:raise ValueError('Source selection hash mismatch')
    for filename,digest in report['files'].items():
        if source_hash(root/filename)!=digest:raise ValueError('Prepared file hash mismatch')
    preprocessing=runtime_json((root/'preprocessing.json').read_text());mean=np.array(preprocessing['mean']);scale=np.array(preprocessing['scale'])
    evidence={};counts={};train_y=None;train_groups=None
    for role in ['train','auxiliary','validation','test']:
        data=load_prepared_data(root/f'{role}.manifest.json',expected_role=role)
        x=data.dataset.tensors[0].numpy();y=data.dataset.tensors[1].numpy()
        oo=np.load(root/f'{role}_ordinals.npy');groups=np.load(root/f'{role}_groups.npy',mmap_mode='r')
        rx=np.load(source/f'{role}_features.npy',mmap_mode='r');ry=np.load(source/f'{role}_labels.npy',mmap_mode='r');ro=np.load(source/f'{role}_ordinals.npy',mmap_mode='r')
        if not np.all(ro[1:]>ro[:-1]):raise ValueError('Source ordinal order is invalid')
        positions=np.searchsorted(ro,oo)
        if (positions>=len(ro)).any() or not np.array_equal(ro[positions],oo):raise ValueError('Unknown original ordinal')
        if not np.array_equal(y,ry[positions]):raise ValueError('Source label alignment differs')
        ev=load_split_evidence(root/f'{role}.ids.json',data_path=root/f'{role}.npz',sample_count=len(y))
        original_role='train' if role=='auxiliary' else role
        if tuple(f'Dataset/{original_role}.csv:row:{int(i)}' for i in oo)!=ev.record_ids:raise ValueError('Record lineage mismatch')
        for start in range(0,len(x),20000):
            end=min(start+20000,len(x));expected=((rx[positions[start:end]].astype(float)-mean)/scale).astype(np.float32)
            if not np.array_equal(expected,x[start:end]):raise ValueError('Source feature transformation differs')
            actual=tuple(input_digest(row) for row in x[start:end])
            if actual!=ev.group_ids[start:end] or actual!=tuple(g.decode() for g in groups[start:end]):raise ValueError('Final group IDs differ from actual input')
        evidence[role]=ev;counts[role]=len(y)
        if role=='train':train_y=np.array(y);train_groups=groups
        del data,x,y,rx,ry,ro;gc.collect()
        print(json.dumps({'stage':'verified_source_alignment','role':role,'rows':counts[role]}),flush=True)
    split=check_disjoint_splits(evidence,policy='group')
    del evidence;gc.collect()
    conditions={}
    train_x=np.load(root/'train_features.npy',mmap_mode='r')
    train_ord=np.load(root/'train_ordinals.npy',mmap_mode='r')
    for name,record in report['conditions'].items():
        directory=root/name;assignment=np.load(directory/'assignment.npy');bundle=runtime_json((directory/'federated_bundle.json').read_text());owners={};rows=[]
        if len(assignment)!=len(train_y):raise ValueError('Client assignment is not complete')
        for group,client in zip(train_groups,assignment):
            previous=owners.setdefault(bytes(group),int(client))
            if previous!=client:raise ValueError('Input group crosses clients')
        for client,description in enumerate(bundle['clients']):
            indices=np.flatnonzero(assignment==client)
            data=load_prepared_data(directory/description['manifest_path'],expected_role='train')
            if not len(indices) or not np.array_equal(data.dataset.tensors[0].numpy(),train_x[indices]) or not np.array_equal(data.dataset.tensors[1].numpy(),train_y[indices]):
                raise ValueError('Client rows differ from shared selected population')
            ev=load_split_evidence(directory/description['evidence_path'],data_path=data.manifest.data_path,sample_count=len(indices))
            if (ev.record_ids!=tuple(f'Dataset/train.csv:row:{int(i)}' for i in train_ord[indices]) or
                    ev.group_ids!=tuple(g.decode() for g in train_groups[indices])):
                raise ValueError('Client source lineage differs from shared population')
            rows.append(len(indices));del data
        if rows!=record['client_rows'] or sum(rows)!=counts['train']:raise ValueError('Client counts differ')
        conditions[name]=rows;print(json.dumps({'stage':'verified_clients','condition':name}),flush=True)
        del owners;gc.collect()
    return {'source_alignment_verified':True,'all_role_records_disjoint':split['declared_record_ids_disjoint'],
        'all_role_groups_disjoint':split['declared_group_ids_disjoint'],'all_client_arrays_verified':True,
        'roles':counts,'conditions':conditions,'preparation_report_sha256':source_hash(root/'preparation_report.json'),
        'source_selection_sha256':source_hash(source/'selection.json'),'test_scored':False,
        'seconds':perf_counter()-started,'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
