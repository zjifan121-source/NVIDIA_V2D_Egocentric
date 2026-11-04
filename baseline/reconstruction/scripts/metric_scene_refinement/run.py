"""Freeze inputs or run a single independent CPU refinement candidate."""
from pathlib import Path
import argparse
import datetime
import importlib
import signal
import time

from common import freeze, load_frozen, put, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    f = commands.add_parser('freeze')
    f.add_argument('--spec', required=True)
    f.add_argument('--output', required=True)
    for name in ['validate', 'run']:
        p = commands.add_parser(name)
        p.add_argument('--manifest', required=True)
        if name == 'run':
            p.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'freeze':
        spec = freeze(args.spec, args.output)
        print(f'Frozen {len(spec["file_sha256"])} files; {spec["frame_count"]} frames')
        return
    spec = load_frozen(args.manifest)
    if args.command == 'validate':
        print('CPU input and provenance validation passed; no fit started')
        return
    output = Path(args.output).resolve()
    for p in spec['inputs'].values():
        q = Path(p)
        if output == q or output.is_relative_to(q) or q.is_relative_to(output):
            raise ValueError('Output must be independent of every input path')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = dict(state='RUNNING', started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   kind=spec['kind'], manifest_sha256=sha(args.manifest), GPU_calls=0, GT_reads=0)
    put(output / 'EXECUTION.json', receipt)
    def timeout(*_):
        raise TimeoutError('CPU wall-clock budget exceeded; no automatic retry')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(int(spec['parameters']['maximum_seconds']))
    try:
        module = importlib.import_module(spec['kind'])
        result = module.run(spec, output)
        receipt.update(state=result['state'], result=result, exit_code=0,
                       production_adopted=False)
        if result.get('bounds_ok') is False or result['state'] in ('INPUT_GATE_FAILED', 'MULTIVIEW_SHAPE_NOT_ACCEPTED'):
            raise ValueError('Candidate not accepted; diagnostics preserved, no automatic downstream run')
    except BaseException as error:
        receipt.update(state='FAILED_PRESERVED', exit_code=1, error=repr(error))
        raise
    finally:
        signal.alarm(0)
        receipt['seconds'] = time.monotonic() - started
        receipt['ended_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        receipt['output_sha256'] = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*'))
                                    if p.is_file() and p.name != 'EXECUTION.json'}
        put(output / 'EXECUTION.json', receipt)


if __name__ == '__main__':
    main()
