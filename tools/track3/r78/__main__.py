"""Explicit stage commands; no default run and no automatic GPU dispatch."""
import argparse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='stage', required=True)
    stage = commands.add_parser('freeze')
    stage.add_argument('--config', required=True)
    stage.add_argument('--output', required=True)
    for name in ('prepare', 'seed', 'fit', 'track'):
        stage = commands.add_parser(name)
        stage.add_argument('--manifest', required=True)
        stage.add_argument('--output', required=True)
        if name == 'prepare':
            stage.add_argument('--candidate', type=int, choices=(1, 2), required=True)
        if name in ('seed', 'track'):
            stage.add_argument('--weights', required=True)
        if name == 'track':
            stage.add_argument('--selection', required=True)
    stage = commands.add_parser('generate')
    stage.add_argument('--prepared', required=True)
    stage.add_argument('--weights', required=True)
    stage.add_argument('--output', required=True)
    stage = commands.add_parser('select')
    stage.add_argument('--pool', required=True)
    stage.add_argument('--output', required=True)
    args = vars(p.parse_args())
    stage = args.pop('stage')
    if stage == 'freeze':
        from .io import freeze
        freeze(**args)
    elif stage in ('prepare', 'generate'):
        from . import generation
        (generation.prepare if stage == 'prepare' else generation.run)(**args)
    elif stage == 'fit':
        from .fitting import run
        run(**args)
    elif stage == 'select':
        from .selection import run
        run(**args)
    else:
        from .tracking import run
        run(**args)


if __name__ == '__main__':
    main()
