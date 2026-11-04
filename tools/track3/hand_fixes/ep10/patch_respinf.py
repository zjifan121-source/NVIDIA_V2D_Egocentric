_H5_MODE = "respin"
_H5_RESPIN_FREEZE_BLEND = True
_P = os.path.join(os.path.dirname(os.path.abspath(os.environ["H5_PATCH"])), "sym_common.py")
exec(compile(open(_P).read(), _P, "exec"))
