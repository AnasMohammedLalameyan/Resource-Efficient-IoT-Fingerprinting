"""Assemble main.tex from the old report (preamble, front matter, Chapter 2, bibliography)
and the new chapter files in chapters/. Diagram files in diagrams/ are inlined.

Usage: python scripts/build_main.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OLD = ROOT / "Previous implementation report" / "main.tex"
lines = OLD.read_text(encoding="utf-8").splitlines(keepends=True)


def seg(a, b):
    """1-based inclusive line range of the old file."""
    return "".join(lines[a - 1:b])


# ---- sanity anchors in the old file ---------------------------------------
assert lines[322].startswith(r"\chapter[Chapter-2 Literature Survey]"), lines[322]
assert lines[411].strip() == r"\newpage", lines[411]
assert lines[1675].startswith(r"\begin{thebibliography}"), lines[1675]
assert lines[1807].startswith(r"\end{thebibliography}"), lines[1807]

# ---- preamble + front matter (old lines 1-273) ---------------------------------
head = seg(1, 273)

# new abbreviations, inserted after the key on the left (keeps alphabetical order)
NEW_ACR = [
    ("AI", "ARM", "Advanced RISC Machine"),
    ("CPU", "CV", "Cross-Validation"),
    ("IP", "IQR", "Interquartile Range"),
    ("MAC", "MCC", "Matthews Correlation Coefficient"),
    ("mDNS", "ML", "Machine Learning"),
    ("PM", "PR-AUC", "Area Under the Precision--Recall Curve"),
    ("RNN", "ROC-AUC", "Area Under the Receiver Operating Characteristic Curve"),
    ("TLS", "TTL", "Time To Live"),
]
for after, key, longform in NEW_ACR:
    for cmd, indent in (("acrodef", ""), ("acro", "  ")):
        pat = re.compile(r"^(" + re.escape(indent) + r"\\" + cmd + r"\{" + re.escape(after) + r"\}\{[^\n]*\}\n)",
                         re.M)
        m = pat.search(head)
        assert m, (cmd, after)
        head = head[:m.end()] + f"{indent}\\{cmd}{{{key}}}{{{longform}}}\n" + head[m.end():]

# allowed preamble additions (Part 2.6)
head = head.replace(
    "\\usepackage{subcaption}\n",
    "\\usepackage{subcaption}\n\n% Diagrams (block diagrams drawn in TikZ)\n"
    "\\usepackage{tikz}\n"
    "\\usetikzlibrary{shapes.geometric,arrows.meta,positioning,fit,calc}\n", 1)

tikzset = r"""
\AtBeginDocument{\renewcommand{\thelstlisting}{\arabic{lstlisting}}}

% Common styles for the block diagrams
\newsavebox{\diagbox}
\newcommand{\fitdiag}[1]{\savebox{\diagbox}{\setstretch{1.0}#1}%
  \ifdim\wd\diagbox>\textwidth\resizebox{\textwidth}{!}{\usebox{\diagbox}}\else\usebox{\diagbox}\fi}
\tikzset{
  blk/.style={rectangle,rounded corners=3pt,draw=black!60,fill=blue!8,thick,align=center,
              minimum height=0.9cm,inner sep=4pt,font=\small},
  small/.style={font=\footnotesize},
  arr/.style={-{Stealth[length=2.2mm]},thick,draw=black!70},
  arrd/.style={-{Stealth[length=2.2mm]},thick,draw=black!70,dashed}
}
"""
anchor = "\\renewcommand{\\figurename}{Fig}\n"
assert anchor in head
head = head.replace(anchor, anchor + tikzset, 1)


def inline_diagrams(text):
    def rep(m):
        p = ROOT / "diagrams" / m.group(1)
        return p.read_text(encoding="utf-8").rstrip() + "\n"
    return re.sub(r"\\input\{diagrams/([^}]+)\}", rep, text)


def allow_breaks(text):
    """Let long \\texttt{...} names (with underscores) break across lines."""
    def rep(m):
        return "\\texttt{" + m.group(1).replace("\\_", "\\_\\allowbreak{}") + "}"
    return re.sub(r"\\texttt\{([^{}]*)\}", rep, text)


def plain_captions(text):
    """Captions are copied to the lists of figures/tables, which are typeset BEFORE the chapters.
    An \\ac{} inside a caption would therefore consume the first-use expansion there.
    Replace \\ac{X} by plain X inside captions."""
    out, i = [], 0
    while True:
        j = text.find("\\caption{", i)
        if j < 0:
            out.append(text[i:])
            break
        k = j + len("\\caption{")
        depth, m = 1, k
        while depth:
            c = text[m]
            depth += (c == "{") - (c == "}")
            m += 1
        body = re.sub(r"\\ac\{([^{}]*)\}", r"\1", text[k:m - 1])
        out.append(text[i:k] + body + "}")
        i = m
    return "".join(out)


def chapter(n):
    p = ROOT / "chapters" / f"ch{n}.tex"
    if not p.exists():
        return f"%% (chapter {n} not written yet)\n"
    text = p.read_text(encoding="utf-8")
    # do not touch code listings
    parts = re.split(r"(\\begin\{lstlisting\}.*?\\end\{lstlisting\})", text, flags=re.S)
    text = "".join(x if x.startswith("\\begin{lstlisting}") else allow_breaks(x) for x in parts)
    return inline_diagrams(plain_captions(text)) + "\n"


# ---- Chapter 2: copied byte for byte (comment banner + chapter + final \newpage) --
ch2 = seg(320, 412)

# Chapter 2: three sentences edited to match the updated scope (user-approved); all else verbatim
CH2_EDITS = [
    ("the challenges of resource efficiency and edge deployability are directly addressed by the present work;",
     "the challenge of resource efficiency is directly addressed by the present work, and edge deployability is addressed in part (preliminary evidence for an earlier model only);"),
    ("directly motivates the device profiling and anomaly alerting components of the present system.",
     "is relevant background; device profiling and anomaly alerting are outside the scope of the present system."),
    ("by implementing a lightweight Decision Tree classifier whose resource consumption is empirically measured and validated on both a development machine and a Raspberry Pi single-board computer.",
     "by comparing a compact pruned Decision Tree and a Random Forest whose resource consumption is empirically measured on a development machine (G1). Edge validation (G2) is addressed only in part: preliminary Raspberry Pi evidence exists for an earlier model, and the updated models remain to be benchmarked on the board."),
]
ch2_orig = ch2
for old, new in CH2_EDITS:
    assert old in ch2, old[:40]
    ch2 = ch2.replace(old, new, 1)

# ---- bibliography ------------------------------------------------------------
bib_head = seg(1674, 1807)  # "%% REFERENCES" banner up to the last old \bibitem
new_bib = r"""
\bibitem{breiman2001}
L.~Breiman,
``Random forests,''
\textit{Machine Learning}, vol.~45, no.~1, pp.~5--32, 2001.

\bibitem{breiman1984}
L.~Breiman, J.~H.~Friedman, R.~A.~Olshen, and C.~J.~Stone,
\textit{Classification and Regression Trees},
Wadsworth, Belmont, CA, USA, 1984.

\bibitem{pedregosa2011}
F.~Pedregosa et al.,
``Scikit-learn: Machine learning in Python,''
\textit{Journal of Machine Learning Research}, vol.~12, pp.~2825--2830, 2011.

"""

out = [head, chapter(1), ch2]
for n in range(3, 11):
    out.append(chapter(n))
out.append(bib_head)
out.append(new_bib)
out.append("\\end{thebibliography}\n\n\\end{document}\n")
(ROOT / "main.tex").write_text("".join(out), encoding="utf-8")

# ---- verify Chapter 2 and front matter are untouched -----------------------------
new_text = (ROOT / "main.tex").read_text(encoding="utf-8")
assert ch2.strip() in new_text, "Chapter 2 missing"
fm_old = seg(175, 209)  # certificate .. list of tables
assert fm_old in new_text, "front matter changed"
print("main.tex written; chapter 2 and front matter verified identical.")
