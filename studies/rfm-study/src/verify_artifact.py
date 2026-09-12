"""Read-only SHA256 verification of the delivered artifact."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'artifact_manifest.json').read_text())
for name,info in manifest['files'].items():
    path=ROOT/name
    assert path.is_file(),name
    assert path.stat().st_size==info['bytes'],name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==info['sha256'],name
actual={str(f.relative_to(ROOT)).replace('\\','/') for f in ROOT.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc' and f.name!='artifact_manifest.json'}
assert actual==set(manifest['files']),sorted(actual.symmetric_difference(manifest['files']))
print(json.dumps(dict(passed=True,files=len(actual),bytes=sum(r['bytes'] for r in manifest['files'].values()))))
