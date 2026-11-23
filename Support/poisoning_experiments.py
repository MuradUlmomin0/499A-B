"""Sequential CP5 controls, pilot and matched attack matrix; no retuning."""
from Support.Utils.runtime_paths import load_json as runtime_json, load_json_file as runtime_json_file, load_yaml as runtime_yaml, safe_load_yaml as runtime_safe_yaml
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml
from Support.data.latest_audit import source_hash
from Support.Utils.runtime_paths import authenticate_recorded, verify_code
from Support.Utils.poisoning_config import load_poisoning_config

REPOSITORY=Path(__file__).resolve().parents[1]
METHODS=['fedavg','fedmade','temporal']


def case_order():
    cases=[dict(name=f'control_{m}_dirichlet_seed42',method=m,condition='dirichlet',seed=42,control=True) for m in METHODS]
    cases += [dict(name=f'{m}_dirichlet_seed42',method=m,condition='dirichlet',seed=42,control=False) for m in METHODS]
    cases += [dict(name=f'{m}_{c}_seed{s}',method=m,condition=c,seed=s,control=False)
              for m in METHODS for c in ['iid','dirichlet'] for s in [42,43,44] if (c,s)!=('dirichlet',42)]
    return cases


def generate_configs(directory,results):
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=False)
    refs={
        'fedavg':runtime_json((REPOSITORY/'Support/docs/cp2_evidence/verification.json').read_text())['primary_runs'],
        'fedmade':runtime_json((REPOSITORY/'Support/results/cp3_v1/matrix/comparison.json').read_text())['runs'],
        'temporal':runtime_json((REPOSITORY/'Support/results/cp4_v1/matrix/comparison.json').read_text())['runs']}
    folders=dict(fedavg='cp2_v1',fedmade='cp3_v1',temporal='cp4_v1')
    for c in case_order():
        ref=next(r['run_path'] for r in refs[c['method']] if (r['condition'],r['seed'])==(c['condition'],c['seed']))
        ref=Path(ref);ref=ref if ref.is_absolute() else REPOSITORY/ref
        base=REPOSITORY/f"Support/configs/{folders[c['method']]}/{c['condition']}_seed{c['seed']}.yaml"
        cfg=dict(schema_version=1,method=c['method'],experiment=dict(name='cp5_'+c['name'],output_dir=os.path.relpath(Path(results).resolve(),directory)),
            baseline_config_path=os.path.relpath(base,directory),reference_clean_run_path=os.path.relpath(ref,directory),
            attack=dict(type='targeted_label',enabled=not c['control'],target_class=18,destination_class=1,
                malicious_clients=0 if c['control'] else 2,start_round=4,end_round=20,selection_seed_offset=5005))
        path=directory/(c['name']+'.yaml');path.write_text(yaml.safe_dump(cfg,sort_keys=False));load_poisoning_config(path)


def guard(config_dir):
    frozen=REPOSITORY/'Support/docs/cp5_evidence/frozen_manifest.json'
    if not frozen.is_file():raise ValueError('CP5 code/configuration freeze is required')
    v=runtime_json(frozen.read_text())
    for field in ['code_sha256','config_sha256']:
        for p,h in v[field].items():
            if not authenticate_recorded(REPOSITORY/p,h):raise ValueError('Frozen CP5 source/configuration changed: '+p)
    evidence=Path(config_dir)/'preflight.json'
    if not evidence.exists():raise ValueError('Measured CP5 preflight required')
    measured=runtime_json(evidence.read_text())
    if set(measured)!=set(METHODS):raise ValueError('Preflight must cover all three methods')
    for m,record in measured.items():
        config=Path(config_dir)/f'{m}_dirichlet_seed42.yaml'
        if (record['config_sha256']!=source_hash(config) or record['training_performed'] or record['test_scored']
                or record['peak_process_rss_kib']>int(3.5*1024**2)):raise ValueError('Stale/invalid memory preflight')


def _checked(path,control):
    v=runtime_json((Path(path)/'verification.json').read_text())
    if (v['testing_only'] or v['test_scored'] or not v['aggregation_verified'] or not v['delivered_labels_verified']
            or not v['reloaded_validation_exact'] or not v['paired_controls_verified'] or (control and v['zero_attack_matches_clean'] is not True)):
        raise ValueError('Failed CP5 run/control verification')
    return v


def run_matrix(config_dir,output_dir,*,resume=False):
    directory=Path(config_dir).resolve();out=Path(output_dir).resolve();guard(directory)
    out.mkdir(parents=True,exist_ok=resume)
    index=out/'progress.json';entries=runtime_json(index.read_text()) if index.exists() else []
    cases=case_order()
    if [e['name'] for e in entries]!=[c['name'] for c in cases[:len(entries)]] or len(entries)>len(cases):raise ValueError('Invalid saved matrix prefix')
    for i,c in enumerate(cases):
        guard(directory);config=directory/(c['name']+'.yaml');cfg=load_poisoning_config(config)
        if i<len(entries):
            e=entries[i]
            if e['config_sha256']!=source_hash(config) or e['verification_sha256']!=source_hash(Path(e['run_path'])/'verification.json'):raise ValueError('Cached case evidence changed')
            _checked(e['run_path'],c['control']);print('Retained verified '+c['name'],flush=True);continue
        results=Path(cfg['experiment']['output_dir']);before=set(results.glob(cfg['experiment']['name']+'_*'))
        print('Starting '+c['name'],flush=True)
        attempt=1
        while (out/f"{c['name']}_attempt{attempt}.log").exists():attempt+=1
        with (out/f"{c['name']}_attempt{attempt}.log").open('x') as stream:
            subprocess.run([sys.executable,'-m','Support.poisoning_experiments','train-one','--config',str(config)],
                env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'},stdout=stream,stderr=subprocess.STDOUT,check=True)
        fresh=set(results.glob(cfg['experiment']['name']+'_*'))-before
        if len(fresh)!=1:raise RuntimeError('Ambiguous CP5 case identity')
        path=fresh.pop();_checked(path,c['control'])
        entries.append({**c,'config_sha256':source_hash(config),'run_path':str(path),'verification_sha256':source_hash(path/'verification.json')})
        temporary=out/'progress.tmp';temporary.write_text(json.dumps(entries,indent=2)+'\n');temporary.replace(index)
        print('Completed '+c['name']+': '+str(path),flush=True)
    return entries


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    for name in ['train-one','preflight']:
        a=sub.add_parser(name);a.add_argument('--config',type=Path,required=True)
    a=sub.add_parser('configs');a.add_argument('--directory',type=Path,required=True);a.add_argument('--results',type=Path,required=True)
    a=sub.add_parser('matrix');a.add_argument('--config-dir',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.action=='configs':generate_configs(args.directory,args.results)
    elif args.action=='matrix':run_matrix(args.config_dir,args.output,resume=args.resume)
    else:
        from Support.poisoning_runner import preflight,run_verified
        if args.action=='preflight':print(json.dumps(preflight(args.config),indent=2))
        else:run_verified(args.config)


if __name__=='__main__':main()
