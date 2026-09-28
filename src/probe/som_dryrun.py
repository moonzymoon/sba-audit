# -*- coding: utf-8 -*-
"""SOM dry-run pack (task 2): build zip per README ship list, verify by readback.
Red lines excluded by construction: E6_zh:zh_4658table_onlyzh:zh_8166.md, any ledger/credential file,
the manuscript PDF itself."""
import hashlib
import os
import zipfile

ROOT = r"D:\0keyan\gongzuo1\paper 15SCI"
OUT = r"D:\0keyan\gongzuo1\paper 15SCI\04_zh:releasezh_8798\SOM_dryrun.zip"
RED = ("zh:zh_4658table_onlyzh:zh_8166", "zh:zh_4125lunwenzh:zh_5435table", "zh:zh_8481infozh_4871", "zh:zh_1041", "zh:zh_6780", "main.pdf")

ship = []
# src analysis code (exclude __pycache__, probe one-offs stay: they are reproducibility)
for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, "src")):
    dirnames[:] = [d for d in dirnames if d != "__pycache__"]
    for f in filenames:
        if f.endswith((".py", ".md")):
            ship.append(os.path.join(dirpath, f))
# prereg + hashes
for f in os.listdir(os.path.join(ROOT, "01_zh:zh_8028withzh:zh_9766")):
    if f.startswith(("pre-registeredzh:zh_8210", "pre-registered_R")) or "E6_zh:anonymouszh_6262" in f:
        ship.append(os.path.join(ROOT, "01_zh:zh_8028withzh:zh_9766", f))
# results CSV/JSON
rd = os.path.join(ROOT, "02_shiyanjilu", "results")
for f in os.listdir(rd):
    if f.endswith((".csv", ".json", ".txt")):
        ship.append(os.path.join(rd, f))
# REPRODUCE
ship.append(os.path.join(ROOT, "02_shiyanjilu", "REPRODUCE.md"))

# red-line guard
for f in ship:
    low = os.path.basename(f)
    assert not any(r in low for r in ("zh:zh_4658table_onlyzh:zh_8166", "zh:zh_5435table", "zh:zh_8481infozh_4871")), "RED LINE: " + low

ship = sorted(set(os.path.normpath(f) for f in ship))
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

manifest = []
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for f in ship:
        arc = os.path.relpath(f, ROOT).replace("\\", "/")
        z.write(f, arc)
        manifest.append((arc, sha(f)))
    mf = "MANIFEST_SHA256.txt"
    body = "\n".join(h + "  " + a for a, h in manifest)
    z.writestr(mf, body)

# verify readback
bad = 0
with zipfile.ZipFile(OUT) as z:
    names = set(z.namelist())
    for arc, h in manifest:
        data = z.read(arc)
        if hashlib.sha256(data).hexdigest() != h:
            bad += 1
    assert mf in names
n = len(manifest)
size = os.path.getsize(OUT)
print("packed %d files (+manifest), zip %.1f MB, readback mismatches: %d" % (n, size / 1e6, bad))
print("red-line scan of arcnames:")
with zipfile.ZipFile(OUT) as z:
    hits = [a for a in z.namelist() if any(r in a for r in RED)]
print("  hits:", hits if hits else "NONE")
print("OK" if bad == 0 and not hits else "FAIL")
