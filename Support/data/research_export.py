"""Training-only scaling, final-input checks and existing v1 research artifacts."""
import csv
import gc
import json
from pathlib import Path

import numpy as np

from Support.data.contract import ModelDataContract
from Support.data.latest_audit import input_digest, source_hash
from Support.data.latest_pilot import _json
from Support.data.research_selection import allocate_groups


def _write_partition(directory, name, role, client, x, y, ordinals, groups, config, contract, preprocessing, testing_only):
    """Bind original source IDs and final-input group IDs to each NPZ."""
    path=directory/f'{name}.npz'
    with path.open('xb') as stream:
        np.savez(stream,features=x,labels=y,feature_names=np.array(config['feature_names']),class_names=np.array(config['class_names']))
    original_role='train' if role=='auxiliary' else role
    records=[f'Dataset/{original_role}.csv:row:{int(i)}' for i in ordinals]
    group_strings=[g.decode('ascii') if isinstance(g,bytes) else str(g) for g in groups]
    _json(directory/f'{name}.ids.json',{'schema_version':1,'source_namespace':'latest:sha256:'+config['archive_sha256'],
        'data_sha256':source_hash(path),'record_ids':records,'group_ids':group_strings})
    _json(directory/f'{name}.manifest.json',{'schema_version':1,'dataset_name':'private-shield-cp1-v1',
        'partition_name':name,'role':role,'client_id':client,'data_format':'npz','data_path':path.name,
        'preprocessing_path':preprocessing,'contract':contract.to_dict(),'testing_only':testing_only})
    return {'manifest_path':f'{name}.manifest.json','evidence_path':f'{name}.ids.json'}


def export_research(source_dir, config, settings, output_dir):
    """Prepare all six bundles without changing client/server or NPZ schemas.

    Final model-input priority is client training, auxiliary, validation, test.
    Exclude later-role collisions without refill; any mixed-label collision or
    loss of required auxiliary/global class support stops publication.
    """
    source=Path(source_dir);out=Path(output_dir)
    if out.exists():raise FileExistsError('Refusing to overwrite research export')
    selection=json.loads((source/'selection.json').read_text())
    if config.get('archive_sha256',selection['archive_sha256'])!=selection['archive_sha256']:
        raise ValueError('Source archive binding mismatch')
    for filename,digest in selection['files'].items():
        if source_hash(source/filename)!=digest:raise ValueError('Selection array hash mismatch')
    config={**config,'archive_sha256':selection['archive_sha256']}
    testing_only=selection['testing_only'];classes=config['class_names'];names=config['feature_names'];nc=len(classes)
    train=np.load(source/'train_features.npy',mmap_mode='r')
    values=train.astype(np.float64);mean=values.mean(axis=0);scale=values.std(axis=0);scale[scale==0]=1
    del values
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():raise ValueError('Nonfinite scaling statistics')
    out.mkdir(parents=True)
    _json(out/'preprocessing.json',{'schema_version':1,'method':'population_standardization',
        'feature_names':names,'class_names':classes,'mean':mean.tolist(),'scale':scale.tolist(),
        'fit_sample_count':len(train),'fit_role':'selected_client_training_only','testing_only':testing_only})
    contract=ModelDataContract('mlp',len(names),nc,tuple(config['hidden_dims']),tuple(names),tuple(classes),source_hash(out/'preprocessing.json'))
    owner={};arrays={};report={'schema_version':1,'testing_only':testing_only,'source_archive_sha256':selection['archive_sha256'],
        'source_selection_sha256':source_hash(source/'selection.json'),'source_selection':selection,
        'test_scored':False,'provenance_verified':False,'simulated_clients':True,'roles':{},'conditions':{}}
    for role_id,role in enumerate(['train','auxiliary','validation','test']):
        raw=np.load(source/f'{role}_features.npy',mmap_mode='r');y=np.load(source/f'{role}_labels.npy',mmap_mode='r');ordinals=np.load(source/f'{role}_ordinals.npy',mmap_mode='r')
        if len(raw)!=len(y) or len(raw)!=len(ordinals) or len(np.unique(ordinals))!=len(raw):raise ValueError('Misaligned source arrays')
        if role=='auxiliary' and not np.all(np.bincount(y,minlength=nc)==settings['auxiliary_groups_per_class']):
            raise ValueError('Initial auxiliary selection does not match the approved per-class count')
        final=np.lib.format.open_memmap(out/f'{role}_features.npy',mode='w+',dtype=np.float32,shape=raw.shape)
        groups=np.lib.format.open_memmap(out/f'{role}_groups.npy',mode='w+',dtype='S64',shape=(len(raw),))
        keep=np.ones(len(raw),dtype=bool)
        for start in range(0,len(raw),20000):
            end=min(start+20000,len(raw));chunk=((raw[start:end].astype(np.float64)-mean)/scale).astype(np.float32)
            if not np.isfinite(chunk).all():raise ValueError('Nonfinite transformed input')
            final[start:end]=chunk
            for j,row in enumerate(chunk,start):
                digest=input_digest(row);label=int(y[j]);groups[j]=digest
                if not 0<=label<nc:raise ValueError('Invalid class ID')
                previous=owner.get(digest)
                if previous is not None:
                    if previous%nc!=label:raise ValueError('Mixed-label final input group')
                    if previous//nc!=role_id or role=='auxiliary':keep[j]=False
                else:owner[digest]=role_id*nc+label
        final.flush();groups.flush()
        indices=np.flatnonzero(keep)
        if not len(indices):raise ValueError(f'No eligible {role} remains')
        # Compact only when exclusion happened, preserving source row order.
        if not keep.all():
            compact=out/f'{role}_features.compact.npy';np.save(compact,final[indices]);del final
            compact.replace(out/f'{role}_features.npy');final=np.load(out/f'{role}_features.npy',mmap_mode='r')
            compact=out/f'{role}_groups.compact.npy';np.save(compact,groups[indices]);del groups
            compact.replace(out/f'{role}_groups.npy');groups=np.load(out/f'{role}_groups.npy',mmap_mode='r')
        yy=np.array(y[indices]);oo=np.array(ordinals[indices]);counts=np.bincount(yy,minlength=nc)
        if not counts.all():raise ValueError(f'Missing global class after final {role} checks')
        if role=='auxiliary':
            if len(np.unique(groups))!=len(yy):
                raise ValueError('Insufficient distinct final auxiliary groups; no refill')
        np.save(out/f'{role}_labels.npy',yy);np.save(out/f'{role}_ordinals.npy',oo)
        arrays[role]=(final,yy,oo,groups)
        report['roles'][role]={'rows':len(yy),'class_counts':counts.tolist(),
            'distinct_groups':len(np.unique(groups)),'final_input_exclusions':int((~keep).sum())}
        _write_partition(out,role,role,None,final,yy,oo,groups,config,contract,'preprocessing.json',testing_only)
        print(json.dumps({'stage':'final_role_export','role':role,**report['roles'][role]}),flush=True)
    del owner;gc.collect()
    train_x,train_y,train_ord,train_groups=arrays['train'];group_strings=[g.decode('ascii') for g in train_groups]
    with (out/'client_class_distributions.csv').open('x',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['condition','seed','client_id','rows',*classes])
        for condition in ['iid','dirichlet']:
            for seed in settings['partition_and_training_seeds']:
                assignment=allocate_groups(group_strings,train_y,settings['clients'],seed,condition,settings['dirichlet_alpha'])
                name=f'{condition}_seed{seed}';directory=out/name;directory.mkdir();np.save(directory/'assignment.npy',assignment)
                clients=[];counts=[]
                for client in range(settings['clients']):
                    indices=np.flatnonzero(assignment==client);client_id=f'client_{client:03d}'
                    ref=_write_partition(directory,client_id,'train',client_id,train_x[indices],train_y[indices],
                        train_ord[indices],train_groups[indices],config,contract,'../preprocessing.json',testing_only)
                    clients.append({'client_id':client_id,**ref})
                    distribution=np.bincount(train_y[indices],minlength=nc).tolist();counts.append(len(indices))
                    writer.writerow([condition,seed,client_id,len(indices),*distribution])
                _json(directory/'federated_bundle.json',{'schema_version':1,'bundle_name':name,'contract':contract.to_dict(),
                    'split_policy':'group','clients':clients,'validation':{'manifest_path':'../validation.manifest.json','evidence_path':'../validation.ids.json'}})
                report['conditions'][name]={'client_rows':counts,'assignment_sha256':source_hash(directory/'assignment.npy'),
                    'bundle_sha256':source_hash(directory/'federated_bundle.json')}
                print(json.dumps({'stage':'client_export','condition':name,'client_rows':counts}),flush=True)
    report['files']={str(p.relative_to(out)):source_hash(p) for p in sorted(out.rglob('*')) if p.is_file()}
    _json(out/'preparation_report.json',report)
    return report
