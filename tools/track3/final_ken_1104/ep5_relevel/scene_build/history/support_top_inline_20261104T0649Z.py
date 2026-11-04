import re, sys
u = sys.argv[1]; txt = open(u).read()
m = re.search(r"(def Cylinder \"support_0\".*?xformOp:translate = \()([^)]+)(\))", txt, re.S)
x, y, z = [float(v) for v in m.group(2).split(",")]
txt = txt[:m.start(2)] + f"{x}, {y}, {1.0097 - 0.005}" + txt[m.end(2):]
open(u, "w").write(txt); print("support_0 center z", z, "->", 1.0097 - 0.005)
