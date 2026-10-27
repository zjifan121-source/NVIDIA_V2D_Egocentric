"""CoTracker3 twin referee for FoundationPose object tracks. `tracks` needs a GPU and CoTracker3 (see README.md);
`relabel` and `apply` run on CPU in the tools environment (numpy, scipy, Pillow)."""
import argparse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='stage', required=True)

    stage = commands.add_parser('tracks', help='CoTracker3 2D point tracks of one object (GPU)')
    source = stage.add_mutually_exclusive_group(required=True)
    source.add_argument('--frames', help="folder of 000000.png ... (a run folder's frames/)")
    source.add_argument('--video', help='the undistorted episode video the masks were made on')
    stage.add_argument('--masks', required=True, help="the object's mask folder (a run folder's masks/1/)")
    stage.add_argument('--checkpoint', required=True, help='CoTracker3 offline checkpoint (scaled_offline.pth)')
    stage.add_argument('--output', required=True, help='tracks .npz')

    stage = commands.add_parser('relabel', help='referee one object run (CPU)')
    stage.add_argument('--run', dest='run_dir', required=True,
                       help='object run folder (poses/, depth/, intrinsics_stable.json, run.json)')
    stage.add_argument('--tracks', required=True, help='npz from `tracks`')
    stage.add_argument('--output', required=True, help='new folder for the corrected poses (must not exist)')
    stage.add_argument('--poses', default='poses', help='pose folder inside the run (default: raw poses/)')
    stage.add_argument('--mesh', help="mesh the poses refer to (default: the run's own, see referee.find_mesh)")
    stage.add_argument('--anchor', default='0', choices=['0', 'ref', 'majority'],
                       help="'0' (default, measured) keeps frame 0's branch; 'ref' the run's registration frame's; "
                            "'majority' the most common branch (diagnostic only, see README)")
    stage.add_argument('--margin', type=float, default=0.6,
                       help="accept a twin when its median error is below margin x the runner-up's (default 0.6)")
    stage.add_argument('--min-pts', dest='min_pts', type=int, default=6,
                       help='lifted points needed per query frame and per frame pair (default 6)')

    stage = commands.add_parser('apply', help='referee every object of a namespace into a new namespace (CPU)')
    stage.add_argument('--outputs', required=True, help='absolute path of baseline/reconstruction/data/outputs')
    stage.add_argument('--source', required=True, help='source namespace (never modified)')
    stage.add_argument('--target', required=True, help='new namespace name')
    stage.add_argument('--tracks-dir', dest='tracks_dir', required=True, help='folder of ep<EP>_<object>.npz')
    stage.add_argument('--episodes', type=int, nargs='+', help='only these episodes (default: all)')
    stage.add_argument('--skip', nargs='*', default=[],
                       help='more objects to leave unchanged, added to blue_cup beige_cup white_pot_lid')
    stage.add_argument('--no-default-skip', dest='default_skip', action='store_false',
                       help='do not leave the cups and the pot lid unchanged (measured to hurt, see README)')
    stage.add_argument('--anchor', default='0', choices=['0', 'ref'],
                       help="'0' (default, measured) or 'ref' (the run's registration frame; not measured on scores)")

    args = vars(p.parse_args())
    stage = args.pop('stage')
    if stage == 'tracks':
        from .tracks import run
    elif stage == 'relabel':
        from .referee import run
    else:
        from .apply import SKIP_DEFAULT, run
        args['skip'] = tuple(dict.fromkeys((SKIP_DEFAULT if args.pop('default_skip') else ()) + tuple(args['skip'])))
    run(**args)


if __name__ == '__main__':
    main()
