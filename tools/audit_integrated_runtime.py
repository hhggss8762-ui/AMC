"""Record unchanged detector files/dependencies and exact shipped AMC/doc files."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
IMAGE = 'weitaoccc/signal-recognition:0.1.0-gpu'
BASE = 'umdetree/signal-detector:0.2.1'
CODE = r'''
import hashlib, importlib.metadata, json, pathlib, sys
files = [pathlib.Path('/app/config.py'), pathlib.Path('/models/model.yaml'), pathlib.Path('/models/localization.pth')]
for directory in ('/app/app', '/app/utils'):
    files.extend(pathlib.Path(directory).rglob('*.py'))
def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()
extra = list(pathlib.Path('/app/amc').glob('*.py'))
extra.extend(p for p in pathlib.Path('/app/docs').rglob('*') if p.is_file())
extra.extend(pathlib.Path(p) for p in ('/app/README.md','/app/docker-compose.yml','/app/examples/client_amc_batch.py') if pathlib.Path(p).is_file())
print(json.dumps({'python':sys.version, 'original_files':{str(p):sha(p) for p in sorted(files)},
    'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
    'added_files':{str(p):sha(p) for p in sorted(extra)}}))
'''


def command(*args):
    return subprocess.check_output(['docker', *args], text=True, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    base_inspect = json.loads(command('image','inspect',BASE))[0]
    image_inspect = json.loads(command('image','inspect',IMAGE))[0]
    baseline = json.loads(command('run','--rm','--network','none','--entrypoint','python',BASE,'-c',CODE))
    integrated = json.loads(command('run','--rm','--network','none','--entrypoint','python',IMAGE,'-c',CODE))
    assert baseline['original_files'] == integrated['original_files'], 'Original detector files changed'
    assert baseline['packages'] == integrated['packages'], 'Inherited dependencies changed'
    base_layers = base_inspect['RootFS']['Layers']
    assert image_inspect['RootFS']['Layers'][:len(base_layers)] == base_layers
    expected = {f'/app/amc/{p.name}':p for p in [
        ROOT/'src/amc'/name for name in ('__init__.py','device.py','model.py','inference.py','prepared_inference.py','service.py','integrated.py')
    ]}
    for relative in ('README.md','AMC_API.md','单信号IQ预处理参考.md',
                     'detector/README.md','detector/api.md','detector/BETA.md','detector/SOURCE.md'):
        expected['/app/docs/'+relative] = ROOT/'docs'/relative
    expected.update({'/app/README.md':ROOT/'docker/IMAGE_README.md',
                     '/app/docker-compose.yml':ROOT/'docker-compose.yml',
                     '/app/examples/client_amc_batch.py':ROOT/'tools/client_amc_batch.py'})
    for target, source in expected.items():
        assert integrated['added_files'][target] == hashlib.sha256(source.read_bytes()).hexdigest(), target
    report = {'base_image_id':base_inspect['Id'], 'image_id':image_inspect['Id'],
              'base_layers_preserved':True, 'original_files_unchanged':True,
              'original_file_count':len(baseline['original_files']),
              'all_dependency_versions_unchanged':True, 'package_count':len(baseline['packages']),
              'shipped_sources_docs_match':True, 'shipped_files_checked':len(expected),
              'image_size_delta_bytes':image_inspect['Size']-base_inspect['Size'],
              'runtime':integrated, 'base_inspect':base_inspect, 'image_inspect':image_inspect}
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key:value for key,value in report.items() if key not in ('runtime','base_inspect','image_inspect')}, indent=2))


if __name__ == '__main__':
    main()
