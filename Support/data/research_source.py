"""Streaming original-role selection for the approved CP1 research population.

Raw arrays are intermediate files, not runner inputs. All source rows are
fingerprinted before filtering. Test contents never select training/auxiliary.
"""
import hashlib
import heapq
import json
from pathlib import Path
import tempfile
from zipfile import ZipFile

import numpy as np
import pandas as pd

from Support.data.latest_audit import ROLES, _sort, source_hash
from Support.data.research_selection import proportional_quotas


def frames(config, role):
    """Yield finite float32 features, class IDs and original zero-based ordinals."""
    names=config['feature_names']; mapping={n:i for i,n in enumerate(config['class_names'])}
    with ZipFile(config['archive_path']) as archive:
        member=f'Dataset/{role}.csv'
        if archive.namelist().count(member)!=1: raise ValueError('Missing/duplicate source member')
        offset=0
        with archive.open(member) as stream:
            for frame in pd.read_csv(stream,dtype={n:np.float32 for n in names},chunksize=config['chunk_rows'],
                    skip_blank_lines=False,encoding='utf-8-sig',on_bad_lines='error'):
                if list(frame.columns)!=[*names,'label'] or not isinstance(frame.index,pd.RangeIndex):
                    raise ValueError('Unexpected source schema')
                x=frame[names].to_numpy(dtype='<f4',copy=True); x[x==0]=0
                y=frame['label'].map(mapping)
                if not np.isfinite(x).all() or y.isna().any(): raise ValueError('Nonfinite input or unknown label')
                yield x,y.to_numpy(dtype=np.int64),np.arange(offset,offset+len(x),dtype=np.int64)
                offset+=len(x)
        if offset!=config['expected_rows'][role]: raise ValueError('Source row count mismatch')


def _offer(heap,cap,item):
    if cap==0:return
    if len(heap)<cap:heapq.heappush(heap,item)
    elif item[0]>heap[0][0]:heapq.heapreplace(heap,item)


def _save_arrays(output,role,x,y,ordinals):
    for name,value in [('features',x),('labels',y),('ordinals',ordinals)]:
        np.save(output/f'{role}_{name}.npy',value,allow_pickle=False)


def prepare_source(config, output_dir, work_dir, training_rows, auxiliary_per_class, seed):
    """Select exact proportional row quotas after whole-group auxiliary removal."""
    output=Path(output_dir)
    if output.exists():raise FileExistsError('Refusing to overwrite research selection')
    if source_hash(config['archive_path'])!=config['archive_sha256']:raise ValueError('Source hash mismatch')
    if type(auxiliary_per_class) is not int or auxiliary_per_class<1:raise ValueError('Invalid auxiliary size')
    output.mkdir(parents=True)
    classes=config['class_names']; names=config['feature_names']; nclasses=len(classes)
    counts=np.zeros(nclasses,dtype=np.int64)
    masks={r:np.zeros(config['expected_rows'][r],dtype=bool) for r in ROLES}
    auxiliary=[[] for _ in classes]
    with tempfile.TemporaryDirectory(prefix='research_source_',dir=work_dir) as tmp:
        scratch=Path(tmp); raw=scratch/'raw.tsv'; ordered=scratch/'ordered.tsv'
        with raw.open('x',buffering=1024**2) as target:
            for role_id,role in enumerate(ROLES):
                for x,y,ordinals in frames(config,role):
                    for row,label,ordinal in zip(x,y,ordinals):
                        digest=hashlib.sha256(row.tobytes()).hexdigest()
                        target.write(f'{digest}\t{role_id}\t{int(label):03d}\t{int(ordinal)}\n')
                print(json.dumps({'stage':'research_source_hashes','role':role}),flush=True)
        _sort(raw,ordered,scratch)
        previous=None; owner=None; group_label=None; train_count=0; first=None; consumed=0
        def finish():
            if previous is None or owner!=0:return
            rank=int.from_bytes(hashlib.sha256(f'auxiliary:{seed}:{group_label}:{previous}'.encode()).digest(),'big')
            _offer(auxiliary[group_label],auxiliary_per_class,(-rank,previous,train_count,first))
        with ordered.open() as stream:
            for line in stream:
                digest,r,y,ordinal=line.rstrip('\n').split('\t');r,y,ordinal=int(r),int(y),int(ordinal)
                if (len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest) or
                        not 0<=r<3 or not 0<=y<nclasses or not 0<=ordinal<len(masks[ROLES[r]]) or
                        previous is not None and digest<previous):raise ValueError('Invalid sorted source signature')
                if digest!=previous:
                    finish();previous=digest;owner=r;group_label=y;train_count=0;first=None
                if y!=group_label:raise ValueError('Mixed-label raw input group')
                if r==owner:masks[ROLES[r]][ordinal]=True
                if r==0:
                    counts[y]+=1;train_count+=1;first=ordinal if first is None else min(first,ordinal)
                consumed+=1
            finish()
        if consumed!=sum(config['expected_rows'].values()):raise RuntimeError('Sorted row count mismatch')
    if any(len(h)!=auxiliary_per_class for h in auxiliary):raise ValueError('Insufficient auxiliary groups')
    aux={item[1]:(y,item[3]) for y,heap in enumerate(auxiliary) for item in heap}
    removed=np.array([sum(item[2] for item in heap) for heap in auxiliary])
    available=counts-removed; quotas=proportional_quotas(available,training_rows)
    if (quotas==0).any():raise ValueError('Proportional sample loses a global training class')
    pools=[[] for _ in classes]; aux_rows={}
    for x,y,ordinals in frames(config,'train'):
        for row,label,ordinal in zip(x,y,ordinals):
            label,ordinal=int(label),int(ordinal);digest=hashlib.sha256(row.tobytes()).hexdigest()
            if digest in aux:
                if ordinal==aux[digest][1]:aux_rows[ordinal]=(row.copy(),label)
                continue
            rank=int.from_bytes(hashlib.blake2b(f'{seed}:train:{ordinal}'.encode(),digest_size=16).digest(),'big')
            _offer(pools[label],int(quotas[label]),(-rank,ordinal,row.copy(),label))
    selected=sorted((item for heap in pools for item in heap),key=lambda item:item[1])
    if len(selected)!=training_rows or len(aux_rows)!=len(aux):raise ValueError('Selected row count mismatch')
    _save_arrays(output,'train',np.stack([i[2] for i in selected]),
                 np.array([i[3] for i in selected],dtype=np.int64),np.array([i[1] for i in selected],dtype=np.int64))
    del selected,pools
    ordinals=sorted(aux_rows)
    _save_arrays(output,'auxiliary',np.stack([aux_rows[i][0] for i in ordinals]),
                 np.array([aux_rows[i][1] for i in ordinals],dtype=np.int64),np.array(ordinals,dtype=np.int64))
    for role in ROLES[1:]:
        n=int(masks[role].sum());position=0
        arrays={key:np.lib.format.open_memmap(output/f'{role}_{key}.npy',mode='w+',dtype=dtype,shape=shape)
                for key,dtype,shape in [('features',np.float32,(n,len(names))),('labels',np.int64,(n,)),('ordinals',np.int64,(n,))]}
        for x,y,ords in frames(config,role):
            keep=masks[role][ords];size=int(keep.sum());end=position+size
            arrays['features'][position:end]=x[keep];arrays['labels'][position:end]=y[keep];arrays['ordinals'][position:end]=ords[keep]
            position=end
        for a in arrays.values():a.flush()
        if position!=n:raise RuntimeError('Holdout export count mismatch')
    if source_hash(config['archive_path'])!=config['archive_sha256']:raise ValueError('Source changed during selection')
    report={'archive_sha256':config['archive_sha256'],'seed':seed,'selected_rows':training_rows,
            'eligible_rows':{r:int(m.sum()) for r,m in masks.items()},'source_training_class_counts':counts.tolist(),
            'auxiliary_groups_per_class':auxiliary_per_class,'auxiliary_removed_training_rows':removed.tolist(),
            'post_auxiliary_training_class_counts':available.tolist(),'quotas':quotas.tolist(),
            'auxiliary_groups':sorted(aux),'test_scored':False,'testing_only':bool(config.get('testing_only',False)),
            'files':{p.name:source_hash(p) for p in sorted(output.glob('*.npy'))}}
    (output/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'stage':'research_selection_complete','quotas':quotas.tolist()}),flush=True)
    return report
