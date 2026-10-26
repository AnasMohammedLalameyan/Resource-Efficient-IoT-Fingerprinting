"""Render selected PDF pages side by side into one PNG for a quick visual check.
Usage: python scripts/preview_pages.py OUT.png DPI page [page ...]
"""
import sys
import pymupdf

out_path, dpi, pages = sys.argv[1], int(sys.argv[2]), [int(p) for p in sys.argv[3:]]
doc = pymupdf.open("main.pdf")
pix = [doc[p - 1].get_pixmap(dpi=dpi) for p in pages]
w = sum(p.width for p in pix)
h = max(p.height for p in pix)
canvas = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, w, h), False)
canvas.set_rect(canvas.irect, (255, 255, 255))
x = 0
for p in pix:
    p = pymupdf.Pixmap(pymupdf.csRGB, p)
    p.set_origin(x, 0)
    canvas.copy(p, p.irect)
    x += p.width
canvas.save(out_path)
