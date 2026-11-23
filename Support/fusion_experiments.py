"""Sequential CP6 controls, pilot and matched attack matrix; no retuning."""
from Support.Utils.runtime_paths import load_json as runtime_json, load_json_file as runtime_json_file, load_yaml as runtime_yaml, safe_load_yaml as runtime_safe_yaml
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml
from Support.data.latest_audit import source_hash
from Support.Utils.runtime_paths import packaged, historical_inventory, authenticate_recorded, verify_code
from Support.Utils.fusion_config import load_fusion_config

REPOSITORY=Path(__file__).resolve().parents[1]
METHODS=['fedavg','fedmade','temporal']


def case_order():
    cases=[dict(name=f'control_lambda{lam}_{attack}_dirichlet_seed42',condition='dirichlet',seed=42,control=True,lambda_value=lam,attacked=attack=='poisoned',repeat=False)
        for lam in [0,1] for attack in ['clean','poisoned']]
    cases += [dict(name=f'fused_{attack}_{condition}_seed{seed}',condition=condition,seed=seed,control=False,lambda_value=.5,attacked=attack=='poisoned',repeat=False)
        for condition in ['iid','dirichlet'] for seed in [42,43,44] for attack in ['clean','poisoned']]
    cases.append(dict(name='repeat_fused_poisoned_dirichlet_seed42',condition='dirichlet',seed=42,control=False,lambda_value=.5,attacked=True,repeat=True))
    return cases


def generate_configs(directory,results):
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=False)
    refs={m:runtime_json((REPOSITORY/f'Support/results/{folder}/matrix/comparison.json').read_text())['runs'] for m,folder in [('fedmade','cp3_v1'),('temporal','cp4_v1')]}
    attacked=runtime_json((REPOSITORY/'Support/docs/cp5_evidence/primary_registry.json').read_text())
    for c in case_order():
        condition,seed=c['condition'],c['seed'];ref=next(r['run_path'] for r in refs['fedmade'] if (r['condition'],r['seed'])==(condition,seed))
        endpoint=None
        if c['control']:
            method='fedmade' if c['lambda_value']==0 else 'temporal'
            pool=attacked if c['attacked'] else refs[method]
            endpoint=next(r['run_path'] for r in pool if (r['condition'],r['seed'])==(condition,seed) and (not c['attacked'] or (r['method']==method and not r['control'])))
        relative=lambda p:os.path.relpath(Path(p).resolve(),directory) if p is not None else None
        cfg=dict(schema_version=1,method='fusion',experiment=dict(name='cp6_'+c['name'],output_dir=relative(results)),
            baseline_config_path=relative(REPOSITORY/f'Support/configs/cp3_v1/{condition}_seed{seed}.yaml'),
            temporal_config_path=relative(REPOSITORY/f'Support/configs/cp4_v1/{condition}_seed{seed}.yaml'),
            reference_clean_run_path=relative(ref),endpoint_reference_run_path=relative(endpoint),lambda_value=c['lambda_value'],
            attack=dict(type='targeted_label',enabled=c['attacked'],target_class=18,destination_class=1,malicious_clients=2 if c['attacked'] else 0,start_round=4,end_round=20,selection_seed_offset=5005))
        path=directory/(c['name']+'.yaml');path.write_text(yaml.safe_dump(cfg,sort_keys=False));load_fusion_config(path)


def _files(directory):
    return {str(p.resolve()) for p in Path(directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts}


def required_inventory(config_dir):
    """Derive the required inventory independently of the supplied freeze maps."""
    directory=Path(config_dir).resolve()
    if packaged(REPOSITORY):
        frozen=historical_inventory(6)
        if directory==Path(frozen['config_directory']):
            return {k:set(frozen[k]) for k in ['code_sha256','config_sha256','input_sha256','reference_sha256']}
    code={str(p.resolve()) for p in (REPOSITORY/'Support').rglob('*.py') if not any(k in p.parts for k in ['results','docs','__pycache__'])}
    code|={str(REPOSITORY/p) for p in ['main.py','Requirements.txt','PROJECT_CONTEXT.md']}
    configs={str(directory/(c['name']+'.yaml')) for c in case_order()}
    inputs={str(directory/'preflight.json')};references=set()
    for path in list(configs):
        cfg=load_fusion_config(path);base=cfg['_baseline']
        configs.update([cfg['baseline_config_path'],cfg['temporal_config_path']])
        bundle=Path(base['dataset']['bundle_path']);inputs.add(str(bundle));inputs.add(str(bundle.parent.parent/'preparation_report.json'));raw=runtime_json(bundle.read_text())
        for item in [*raw['clients'],raw['validation']]:
            for field in ['manifest_path','evidence_path']:inputs.add(str((bundle.parent/item[field]).resolve()))
            manifest=(bundle.parent/item['manifest_path']).resolve();m=runtime_json(manifest.read_text())
            for field in ['data_path','preprocessing_path']:inputs.add(str((manifest.parent/m[field]).resolve()))
        manifest=Path(base['fedmade']['auxiliary_manifest_path']);inputs.add(str(manifest));inputs.add(base['fedmade']['auxiliary_evidence_path'])
        m=runtime_json(manifest.read_text())
        for field in ['data_path','preprocessing_path']:inputs.add(str((manifest.parent/m[field]).resolve()))
        for field in ['reference_clean_run_path','endpoint_reference_run_path']:
            if cfg[field] is not None:references.update(_files(cfg[field]))
    for relative in ['Support/results/cp3_v1/matrix/comparison.json','Support/results/cp4_v1/matrix/comparison.json',
        'Support/docs/cp5_evidence/primary_registry.json','Support/docs/cp5_evidence/comparison.json','Support/docs/cp5_evidence/paired_per_class.csv']:
        references.add(str(REPOSITORY/relative))
    return dict(code_sha256=code,config_sha256=configs,input_sha256=inputs,reference_sha256=references)


def validate_case(config,case):
    """Bind a case's scientific identity to approved baseline and endpoint paths."""
    condition,seed=case['condition'],case['seed'];base=config['_baseline']
    attack=dict(type='targeted_label',enabled=case['attacked'],target_class=18,destination_class=1,malicious_clients=2 if case['attacked'] else 0,start_round=4,end_round=20,selection_seed_offset=5005)
    if config['attack']!=attack or Path(config['experiment']['output_dir'])!=REPOSITORY/'Support/results/cp6_v1/runs':raise ValueError('Case attack/output differs from approved protocol')
    if (config['lambda_value']!=case['lambda_value'] or config['attack']['enabled']!=case['attacked']
        or base['experiment']['seed']!=seed or config['experiment']['name']!='cp6_'+case['name']
        or Path(config['baseline_config_path'])!=REPOSITORY/f'Support/configs/cp3_v1/{condition}_seed{seed}.yaml'
        or Path(config['temporal_config_path'])!=REPOSITORY/f'Support/configs/cp4_v1/{condition}_seed{seed}.yaml'):
        raise ValueError('Case configuration differs from frozen scientific identity')
    refs=runtime_json((REPOSITORY/'Support/results/cp3_v1/matrix/comparison.json').read_text())['runs']
    expected_clean=next(r['run_path'] for r in refs if (r['condition'],r['seed'])==(condition,seed))
    if Path(config['reference_clean_run_path'])!=Path(expected_clean):raise ValueError('Case clean reference mismatch')
    expected_endpoint=None
    if case['control']:
        method='fedmade' if case['lambda_value']==0 else 'temporal'
        if case['attacked']:
            refs=runtime_json((REPOSITORY/'Support/docs/cp5_evidence/primary_registry.json').read_text())
            expected_endpoint=next(r['run_path'] for r in refs if r['method']==method and not r['control'] and (r['condition'],r['seed'])==(condition,seed))
        else:
            folder='cp3_v1' if method=='fedmade' else 'cp4_v1';refs=runtime_json((REPOSITORY/f'Support/results/{folder}/matrix/comparison.json').read_text())['runs']
            expected_endpoint=next(r['run_path'] for r in refs if (r['condition'],r['seed'])==(condition,seed))
    if config['endpoint_reference_run_path']!=expected_endpoint:raise ValueError('Case endpoint reference mismatch')


def freeze(config_dir,path):
    directory=Path(config_dir).resolve();inventory=required_inventory(directory)
    for c in case_order():validate_case(load_fusion_config(directory/(c['name']+'.yaml')),c)
    record=dict(protocol_version='cp6_equal_blend_v1',config_directory=str(directory),user_approved=True,test_scored=False,
        **{field:{p:source_hash(p) for p in sorted(paths)} for field,paths in inventory.items()})
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('x') as f:json.dump(record,f,indent=2)
    return record


def guard(config_dir):
    frozen=REPOSITORY/'Support/docs/cp6_evidence/frozen_manifest.json'
    if not frozen.is_file():raise ValueError('CP6 source/configuration freeze required')
    v=runtime_json(frozen.read_text());directory=Path(config_dir).resolve()
    if v.get('config_directory')!=str(directory) or v.get('protocol_version')!='cp6_equal_blend_v1' or v.get('user_approved') is not True or v.get('test_scored') is not False:raise ValueError('Invalid freeze identity')
    for field,required in required_inventory(directory).items():
        if set(v.get(field,{}))!=required or not required:raise ValueError('Incomplete/extra frozen inventory: '+field)
        for p,h in v[field].items():
            if not authenticate_recorded(p,h):raise ValueError('Frozen source/config/input/reference changed: '+p)
    for case in case_order():validate_case(load_fusion_config(directory/(case['name']+'.yaml')),case)
    evidence=directory/'preflight.json'
    if not evidence.exists():raise ValueError('Measured CP6 preflight required')
    measured=runtime_json(evidence.read_text());config=directory/'fused_poisoned_dirichlet_seed42.yaml'
    if (measured['config_sha256']!=source_hash(config) or measured['training_performed'] or measured['test_scored']
        or not measured['both_components_initialized'] or measured['peak_process_rss_kib']>int(3.5*1024**2)):raise ValueError('Stale/invalid memory preflight')


def artifact_hashes(path):return {p:source_hash(p) for p in sorted(_files(path))}


def check_cached(entry,case,config,manifest):
    if any(entry.get(k)!=v for k,v in case.items()):raise ValueError('Cached case metadata differs')
    path=Path(entry['run_path']);cfg=load_fusion_config(config)
    if load_fusion_config(path/'config.yaml')!=cfg:raise ValueError('Cached run configuration differs')
    if entry['config_sha256']!=source_hash(config) or entry['verification_sha256']!=source_hash(path/'verification.json'):raise ValueError('Cached case digest differs')
    if entry['artifact_manifest_sha256']!=source_hash(manifest) or runtime_json(Path(manifest).read_text())!=artifact_hashes(path):raise ValueError('Cached artifacts changed')
    _checked(path,case['control'])


def check_repeat(entries):
    repeats=[e for e in entries if e['repeat']]
    if not repeats:return
    original=next(e for e in entries if e['name']=='fused_poisoned_dirichlet_seed42');repeat=repeats[0]
    left,right=Path(original['run_path']),Path(repeat['run_path'])
    if (runtime_json((left/'verification.json').read_text())['final_state_sha256']!=runtime_json((right/'verification.json').read_text())['final_state_sha256']
        or (left/'validation_metrics.json').read_bytes()!=(right/'validation_metrics.json').read_bytes()):raise ValueError('Full repeat differs')


def _checked(path,control):
    v=runtime_json((Path(path)/'verification.json').read_text())
    if (v['testing_only'] or v['test_scored'] or not v['aggregation_verified'] or not v['delivered_labels_verified']
            or not v['reloaded_validation_exact'] or not v['paired_controls_verified'] or (control and v['endpoint_matches_reference'] is not True)):
        raise ValueError('Failed CP6 run/control verification')
    return v


def run_matrix(config_dir,output_dir,*,resume=False):
    directory=Path(config_dir).resolve();out=Path(output_dir).resolve();guard(directory)
    out.mkdir(parents=True,exist_ok=resume)
    index=out/'progress.json';entries=runtime_json(index.read_text()) if index.exists() else []
    cases=case_order()
    if [e['name'] for e in entries]!=[c['name'] for c in cases[:len(entries)]] or len(entries)>len(cases):raise ValueError('Invalid saved matrix prefix')
    for i,c in enumerate(cases):
        guard(directory);config=directory/(c['name']+'.yaml');cfg=load_fusion_config(config)
        if i<len(entries):
            e=entries[i];check_cached(e,c,config,out/(c['name']+'_artifacts.json'))
            check_repeat(entries[:i+1]);print('Retained verified '+c['name'],flush=True);continue
        validate_case(cfg,c)
        results=Path(cfg['experiment']['output_dir']);before=set(results.glob(cfg['experiment']['name']+'_*'))
        print('Starting '+c['name'],flush=True)
        attempt=1
        while (out/f"{c['name']}_attempt{attempt}.log").exists():attempt+=1
        with (out/f"{c['name']}_attempt{attempt}.log").open('x') as stream:
            subprocess.run([sys.executable,'-m','Support.fusion_experiments','train-one','--config',str(config)],
                env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'},stdout=stream,stderr=subprocess.STDOUT,check=True)
        fresh=set(results.glob(cfg['experiment']['name']+'_*'))-before
        if len(fresh)!=1:raise RuntimeError('Ambiguous CP6 case identity')
        path=fresh.pop();_checked(path,c['control'])
        if load_fusion_config(path/'config.yaml')!=cfg:raise ValueError('Completed run configuration differs')
        manifest=out/(c['name']+'_artifacts.json')
        with manifest.open('x') as stream:json.dump(artifact_hashes(path),stream,indent=2)
        entries.append({**c,'config_sha256':source_hash(config),'run_path':str(path),'verification_sha256':source_hash(path/'verification.json'),
            'artifact_manifest_sha256':source_hash(manifest)})
        check_repeat(entries)
        temporary=out/'progress.tmp';temporary.write_text(json.dumps(entries,indent=2)+'\n');temporary.replace(index)
        print('Completed '+c['name']+': '+str(path),flush=True)
    check_repeat(entries)
    return entries


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    for name in ['train-one','preflight']:
        a=sub.add_parser(name);a.add_argument('--config',type=Path,required=True)
    a=sub.add_parser('configs');a.add_argument('--directory',type=Path,required=True);a.add_argument('--results',type=Path,required=True)
    a=sub.add_parser('freeze');a.add_argument('--config-dir',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('matrix');a.add_argument('--config-dir',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.action=='configs':generate_configs(args.directory,args.results)
    elif args.action=='freeze':freeze(args.config_dir,args.output)
    elif args.action=='matrix':run_matrix(args.config_dir,args.output,resume=args.resume)
    else:
        from Support.fusion_runner import preflight,run_verified
        if args.action=='preflight':print(json.dumps(preflight(args.config),indent=2))
        else:run_verified(args.config)


if __name__=='__main__':main()
