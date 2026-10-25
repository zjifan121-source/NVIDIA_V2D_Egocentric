"""Stage commands of the multi-key-frame pick. `plan` and `pick` are CPU-only; `draw` runs in the reconstruction host
environment; scoring runs inside the FoundationPose runtime with score_fp.py (see README.md)."""
import argparse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='stage', required=True)
    stage = commands.add_parser('plan', help='choose draw frames and held-out frames from a run folder')
    stage.add_argument('--run', dest='run_dir', required=True,
                       help='object run folder with frames/, depth/, masks/1/, intrinsics_stable.json')
    stage.add_argument('--output', required=True, help='plan.json (draws are written next to it)')
    stage.add_argument('--episode', type=int)
    stage.add_argument('--object')
    stage.add_argument('--key-frame', dest='key_frame', type=int,
                       help="frame of the pipeline's own mesh and of the tracking (default: run.json reference_frame)")
    stage.add_argument('--seeds', type=int, nargs='+', default=[1, 2])
    stage.add_argument('--heldout-step', dest='heldout_step', type=int, default=2,
                       help='held-out frames come from every n-th mask (default 2; see README)')
    stage = commands.add_parser('draw', help='SAM 3D draws of a plan (reconstruction host environment, GPU)')
    stage.add_argument('--plan', required=True)
    stage.add_argument('--candidate', action='append', help='only these candidate ids (default: all)')
    stage.add_argument('--no-dev', dest='dev', action='store_false',
                       help="use the images' baked module code instead of this checkout's (default: --dev, as the "
                            'evidence was made)')
    stage.add_argument('--recon-dir', dest='recon_dir', help='baseline/reconstruction to run from (default: this repo)')
    stage = commands.add_parser('pick', help='choose the candidate from score_fp.py output')
    stage.add_argument('--plan', required=True)
    stage.add_argument('--scores', required=True)
    stage.add_argument('--output', required=True)
    args = vars(p.parse_args())
    stage = args.pop('stage')
    if stage == 'plan':
        from .plan import run
    elif stage == 'draw':
        from .draw import run
    else:
        from .pick import run
    run(**args)


if __name__ == '__main__':
    main()
