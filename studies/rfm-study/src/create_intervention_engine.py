from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'src/experiment_v2.py').read_text().replace('from directions_v2 import initial_direction','from directions_v3 import initial_direction')
source=source.replace("ROOT/'src/directions_v2.py'", "ROOT/'src/directions_v2.py',ROOT/'src/directions_v3.py'")
source=source.replace("    (path/'environment.json').write_text", "    if c['kind']=='custom':env['initializer_sha256']=hashlib.sha256((ROOT/'initializations'/c['amount']).read_bytes()).hexdigest()\n    (path/'environment.json').write_text",1)
path=ROOT/'src/experiment_v3.py'
if path.exists():raise FileExistsError(path)
path.write_text(source)
