"""run_posix_glob.py SCRIPT ARGS... : run a Linux-written script on Windows with glob results using '/' separators
(the scripts parse 'sequence_id=([^/]+)' out of glob paths). Nothing else is changed."""
import glob, runpy, sys
_g = glob.glob
glob.glob = lambda *a, **k: [p.replace("\\", "/") for p in _g(*a, **k)]
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
