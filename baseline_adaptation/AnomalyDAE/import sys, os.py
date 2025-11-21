import sys, os
import torch
ok = {"python": sys.version.split()[0], "torch": torch.__version__}
try:
    import dgl; ok["dgl"] = dgl.__version__
except Exception as e:
    ok["dgl_error"] = str(e)
print(ok)
print("Datasets present:", any(f.endswith(".mat") for f in os.listdir(".")))