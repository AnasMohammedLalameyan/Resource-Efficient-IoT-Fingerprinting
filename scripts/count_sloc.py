"""Count SLOC = physical lines - blank lines - comment-only lines - docstring lines
for the 12 core modules. Usage: python scripts/count_sloc.py
"""
import ast
import io
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = ["stages/stage1_extract.py", "stages/stage2_clean_select.py", "stages/stage3_train.py",
         "stages/stage3b_train_rf.py", "stages/stage4_evaluate.py", "stages/stage5_benchmark.py",
         "stages/stage5b_compare.py", "stages/stage5c_resource.py", "stages/stage6_report.py",
         "tuning_protocol.py", "config.py", "io_utils.py"]


def count(path):
    text = (ROOT / path).read_text(encoding="utf-8")
    lines = text.splitlines()
    phys = len(lines)
    blank = sum(1 for l in lines if not l.strip())
    doc = set()
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                doc.update(range(body[0].lineno, body[0].end_lineno + 1))
    comment_only = set()
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT and not lines[tok.start[0] - 1][:tok.start[1]].strip():
            comment_only.add(tok.start[0])
    nonblank_doc = {n for n in doc if lines[n - 1].strip()}
    nonblank_cm = {n for n in comment_only if n not in nonblank_doc}
    return phys, blank, len(nonblank_cm) + len(nonblank_doc), phys - blank - len(nonblank_cm) - len(nonblank_doc)


tot = [0, 0, 0, 0]
for f in FILES:
    r = count(f)
    tot = [a + b for a, b in zip(tot, r)]
    print(f"{f:34s} phys={r[0]:4d} blank={r[1]:3d} comment+doc={r[2]:3d} sloc={r[3]:4d}")
print("TOTAL", tot)
