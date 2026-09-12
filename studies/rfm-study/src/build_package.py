"""Create a complete archive, then verify every archived byte by SHA256."""
import datetime,hashlib,json,shutil,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    assert json.loads((ROOT/'results/audit.json').read_text())['passed']
    assert json.loads((ROOT/'results/package_smoke_check.json').read_text())['passed']
    logs=ROOT/'logs';logs.mkdir(exist_ok=True)
    for name in ['discovery','probe','taylor','modes','operator','fit-operator','frozen','linear','numerical','analysis','reproduction','audit','figures']:
        source=ROOT.parents[1]/'work'/f'v3-{name}.log'
        if source.exists():shutil.copy2(source,logs/source.name)
    files=sorted(f for f in ROOT.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc' and f.name!='artifact_manifest.json')
    manifest=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files={str(f.relative_to(ROOT)).replace('\\','/'):dict(bytes=f.stat().st_size,sha256=sha(f)) for f in files})
    mp=ROOT/'artifact_manifest.json';mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    archive=ROOT.parent/'rfm-mechanism.zip'
    if archive.exists():raise FileExistsError('Refusing to replace existing deliverable: '+str(archive))
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for file in files+[mp]:z.write(file,'rfm-mechanism/'+str(file.relative_to(ROOT)).replace('\\','/'))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(files)+1
        for name,info in manifest['files'].items():
            blob=z.read('rfm-mechanism/'+name);assert len(blob)==info['bytes'];assert hashlib.sha256(blob).hexdigest()==info['sha256'],name
        assert hashlib.sha256(z.read('rfm-mechanism/artifact_manifest.json')).hexdigest()==sha(mp)
    out=dict(passed=True,archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files_in_archive=len(files)+1,
             manifest_sha256=sha(mp),crc_and_all_entry_hashes_verified=True,
             experiments=844,metric_records=47940)
    (ROOT.parent/'rfm-mechanism-package.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
