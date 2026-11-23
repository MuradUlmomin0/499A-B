"""Resolve packaged historical paths locally without rewriting research records.

Only prefixes explicitly listed in runtime_manifest.json are relocated. There
is no fallback to the old project, Downloads or the original dataset directory.
JSON/YAML bytes and their recorded hashes stay unchanged. Archived source is
used only for authenticating historical source inventories; data/model hashes
always refer to the actual packaged files. The current executable code has its
own independent fingerprint inventory.
"""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[2]
_cached=None
_archive_checks={}


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def manifest():
    global _cached
    path=ROOT/'Support/runtime_manifest.json'
    if not path.is_file():return None
    stamp=(path.stat().st_mtime_ns,path.stat().st_size)
    if _cached is None or _cached[0]!=stamp:
        value=json.loads(path.read_text())
        if value.get('schema_version')!=1 or value.get('purpose')!='self_contained_runtime':
            raise ValueError('Invalid runtime package manifest')
        _cached=(stamp,value)
    return _cached[1]


def packaged(repository=None):
    return (repository is None or Path(repository).resolve()==ROOT) and manifest() is not None


def original_member(path):
    parts=list(Path(path).parts)
    if parts and parts[0]=='Support':parts[0]='support'
    if len(parts)>1 and parts[0]=='support' and parts[1]=='Utils':parts[1]='utils'
    if parts==['Requirements.txt']:parts=['requirements.txt']
    return '/'.join(parts)


def _local_relative(path):
    parts=list(Path(path).parts)
    if parts and parts[0]=='support':parts[0]='Support'
    if len(parts)>1 and parts[0]=='Support' and parts[1]=='utils':parts[1]='Utils'
    if parts==['requirements.txt']:parts=['Requirements.txt']
    return Path(*parts)


def relocate(value):
    return _relocate_value(value,manifest())


def _relocate_value(value,info):
    if info is None:return value
    if isinstance(value,str):
        exact=info.get('external_files',{}).get(value)
        if exact is not None:return str(ROOT/exact)
        prefix=info['origin_root']
        if value==prefix:return str(ROOT)
        if value.startswith(prefix+'/'):
            return str(ROOT/_local_relative(value[len(prefix)+1:]))
        if value.startswith('support/'):
            return str(ROOT/_local_relative(value))
        if value=='requirements.txt':return 'Requirements.txt'
        return value
    if isinstance(value,list):return [_relocate_value(v,info) for v in value]
    if isinstance(value,dict):
        converted={}
        for key,item in value.items():
            new_key=_relocate_value(key,info)
            if new_key in converted:raise ValueError('Relocated mapping keys collide')
            converted[new_key]=_relocate_value(item,info)
        return converted
    return value


def load_json(*args,**kwargs):return relocate(json.loads(*args,**kwargs))


def load_json_file(*args,**kwargs):return relocate(json.load(*args,**kwargs))


def load_yaml(*args,**kwargs):
    import yaml
    return relocate(yaml.load(*args,**kwargs))


def safe_load_yaml(*args,**kwargs):
    import yaml
    return relocate(yaml.safe_load(*args,**kwargs))


def verify_code():
    info=manifest()
    if info is None:return
    for relative,expected in info['runtime_sha256'].items():
        p=ROOT/relative
        if not p.is_file() or sha256(p)!=expected:
            raise ValueError('Runtime source changed: '+relative)


def authenticate_recorded(path,expected):
    """Authenticate actual inputs or explicitly archived historical source."""
    path=Path(relocate(str(path)))
    if path.is_file() and sha256(path)==expected:return True
    info=manifest()
    if info is None or not path.is_relative_to(ROOT):return False
    relative=path.relative_to(ROOT)
    allowed=(path.suffix=='.py' or str(relative) in
             {'Requirements.txt','PROJECT_CONTEXT.md','Support/docs/cp7_protocol.md'})
    if not allowed:return False
    verify_code()
    member=original_member(relative)
    for archive_relative in info['source_archives']:
        archive=ROOT/archive_relative;stamp=(archive.stat().st_mtime_ns,archive.stat().st_size)
        if _archive_checks.get(archive_relative)!=stamp:
            if sha256(archive)!=info['assets'][archive_relative]:
                raise ValueError('Historical source archive changed')
            _archive_checks[archive_relative]=stamp
        with ZipFile(archive) as z:
            if member in z.namelist():
                with z.open(member) as stream:
                    if hashlib.file_digest(stream,'sha256').hexdigest()==expected:return True
    return False


def historical_inventory(checkpoint):
    info=manifest()
    if info is None:return None
    relative=f'Support/docs/cp{checkpoint}_evidence/frozen_manifest.json'
    path=ROOT/relative
    if sha256(path)!=info['assets'][relative]:
        raise ValueError('Original frozen manifest changed')
    return load_json(path.read_text())


def verify_bundle():
    """Stream all checksums; do not load large datasets or forward models."""
    verify_code();info=manifest()
    if info is None:raise ValueError('No runtime package manifest')
    for relative,expected in info['assets'].items():
        path=ROOT/relative
        if not path.is_file() or sha256(path)!=expected:
            raise ValueError('Packaged runtime asset changed: '+relative)
    return {'assets_verified':len(info['assets']),'runtime_files_verified':len(info['runtime_sha256']),
            'external_project_required':False,'models_trained':False,'test_scored':False}


if __name__=='__main__':print(json.dumps(verify_bundle(),indent=2))
