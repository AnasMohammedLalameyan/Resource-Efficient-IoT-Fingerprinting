"""Make contact sheets of the result charts for a quick visual check.
Usage: python scripts/contact_sheet.py OUT_PREFIX
"""
import sys
from pathlib import Path
from PIL import Image

files = sorted(Path("Results Figures").glob("r_*.png"))
prefix = sys.argv[1]
per = 6
for k in range(0, len(files), per):
    group = files[k:k + per]
    ims = [Image.open(f).convert("RGB") for f in group]
    w = 760
    ims = [im.resize((w, int(im.height * w / im.width))) for im in ims]
    cols = 2
    rows = (len(ims) + 1) // 2
    rh = [max(im.height for im in ims[r * 2:r * 2 + 2]) for r in range(rows)]
    sheet = Image.new("RGB", (cols * w, sum(rh)), "white")
    y = 0
    for r in range(rows):
        for c, im in enumerate(ims[r * 2:r * 2 + 2]):
            sheet.paste(im, (c * w, y))
        y += rh[r]
    sheet.save(f"{prefix}{k // per}.png")
    print(prefix + str(k // per), [f.name for f in group])
