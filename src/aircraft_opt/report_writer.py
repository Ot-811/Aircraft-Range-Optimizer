"""A very small report builder shared by the main report and the demo report.

A report is a list of blocks (heading, paragraph, bullet list, table, figure). The same list
is written twice:
    report.md   Markdown, readable on GitHub / VS Code
    report.tex  LaTeX, compiled to report.pdf if `pdflatex` is installed (MiKTeX / TeX Live)

Text may use **bold** and `code`. Unicode symbols (², ≤, λ, ...) are translated for LaTeX.
The reports never contain hand-typed numbers: every number comes from the saved results.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

# Unicode -> LaTeX (text mode). Anything not listed and not ASCII is replaced by '?'.
UNICODE_TO_LATEX = {
    "²": r"\textsuperscript{2}", "³": r"\textsuperscript{3}", "⁻": r"\textsuperscript{-}",
    "¹": r"\textsuperscript{1}", "⁰": r"\textsuperscript{0}", "⁴": r"\textsuperscript{4}",
    "⁵": r"\textsuperscript{5}", "⁶": r"\textsuperscript{6}", "⁷": r"\textsuperscript{7}",
    "⁸": r"\textsuperscript{8}", "⁹": r"\textsuperscript{9}",
    "≤": r"$\le$", "≥": r"$\ge$", "≈": r"$\approx$", "×": r"$\times$", "·": r"$\cdot$",
    "−": r"$-$", "±": r"$\pm$", "→": r"$\rightarrow$", "⇒": r"$\Rightarrow$", "√": r"$\surd$",
    "∞": r"$\infty$", "∇": r"$\nabla$", "‖": r"$\|$", "Δ": r"$\Delta$", "λ": r"$\lambda$",
    "μ": r"$\mu$", "ρ": r"$\rho$", "η": r"$\eta$", "ν": r"$\nu$", "°": r"$^\circ$",
    "π": r"$\pi$", "∝": r"$\propto$", "Σ": r"$\Sigma$", "∂": r"$\partial$", "δ": r"$\delta$",
    "ᵢ": r"$_i$", "₀": r"$_0$", "₁": r"$_1$", "ᵀ": r"$^T$", "½": r"$\frac{1}{2}$", "←": r"$\leftarrow$",
    "–": "--", "—": "---", "’": "'", "“": "``", "”": "''", "✓": r"$\checkmark$", "✗": r"$\times$",
}
LATEX_SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                 "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def _latex_plain(text: str) -> str:
    out = []
    for ch in text:
        if ch in LATEX_SPECIAL:
            out.append(LATEX_SPECIAL[ch])
        elif ch in UNICODE_TO_LATEX:
            out.append(UNICODE_TO_LATEX[ch])
        elif ord(ch) < 128:
            out.append(ch)
        else:
            out.append("?")
    return "".join(out)


def latex_inline(text: str) -> str:
    """Escape text for LaTeX and translate **bold** and `code`."""
    parts = re.split(r"(\*\*.+?\*\*|`.+?`)", str(text))
    out = []
    for part in parts:
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            out.append(r"\textbf{" + _latex_plain(part[2:-2]) + "}")
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            out.append(r"\texttt{" + _latex_plain(part[1:-1]) + "}")
        else:
            out.append(_latex_plain(part))
    return "".join(out)


class Report:
    def __init__(self, title: str, subtitle: str = ""):
        self.title = title
        self.subtitle = subtitle
        self.blocks = []

    # ------------------------------------------------------------------ building
    def heading(self, text, level=1):
        self.blocks.append(("heading", level, text))

    def para(self, text):
        self.blocks.append(("para", text))

    def bullets(self, items):
        self.blocks.append(("bullets", list(items)))

    def table(self, header, rows, caption=None):
        self.blocks.append(("table", list(header), [list(map(str, r)) for r in rows], caption))

    def figure(self, path, caption, width=0.9):
        self.blocks.append(("figure", Path(path), caption, width))

    # ------------------------------------------------------------------ Markdown
    def to_markdown(self, folder: Path) -> str:
        lines = [f"# {self.title}", ""]
        if self.subtitle:
            lines += [f"_{self.subtitle}_", ""]
        for b in self.blocks:
            kind = b[0]
            if kind == "heading":
                lines += ["#" * (b[1] + 1) + " " + b[2], ""]
            elif kind == "para":
                lines += [b[1], ""]
            elif kind == "bullets":
                lines += [f"- {item}" for item in b[1]] + [""]
            elif kind == "table":
                header, rows, caption = b[1], b[2], b[3]
                if caption:
                    lines += [f"**{caption}**", ""]
                lines.append("| " + " | ".join(header) + " |")
                lines.append("|" + "---|" * len(header))
                lines += ["| " + " | ".join(r) + " |" for r in rows]
                lines.append("")
            elif kind == "figure":
                rel = Path(os.path.relpath(b[1], folder)).as_posix()
                lines += [f"![{b[2]}]({rel})", "", f"_{b[2]}_", ""]
        return "\n".join(lines)

    # ------------------------------------------------------------------ LaTeX
    def to_latex(self, folder: Path) -> str:
        L = [r"\documentclass[10pt,a4paper]{article}",
             r"\usepackage[T1]{fontenc}", r"\usepackage{lmodern}", r"\usepackage{amssymb}",
             r"\usepackage[margin=2cm]{geometry}", r"\usepackage{graphicx}", r"\usepackage{booktabs}",
             r"\usepackage{float}", r"\usepackage{tabularx}", r"\usepackage{array}",
             r"\setlength{\parskip}{4pt}", r"\setlength{\parindent}{0pt}",
             r"\begin{document}",
             r"\begin{center}{\Large\bfseries " + latex_inline(self.title) + r"}\\[4pt]",
             r"{\small " + latex_inline(self.subtitle) + r"}\end{center}"]
        for b in self.blocks:
            kind = b[0]
            if kind == "heading":
                cmd = {1: "section", 2: "subsection"}.get(b[1], "subsubsection")
                L.append(rf"\{cmd}{{{latex_inline(b[2])}}}")
            elif kind == "para":
                L.append(latex_inline(b[1]) + "\n")
            elif kind == "bullets":
                L.append(r"\begin{itemize}\setlength{\itemsep}{1pt}")
                L += [r"\item " + latex_inline(item) for item in b[1]]
                L.append(r"\end{itemize}")
            elif kind == "table":
                header, rows, caption = b[1], b[2], b[3]
                n = len(header)
                # rough width in characters; wide tables get wrapping columns (tabularx)
                width = sum(max(len(str(r[j])) for r in [header] + rows) for j in range(n)) + 3 * n
                if width > 105:
                    begin = r"\begin{tabularx}{\textwidth}{l" + r">{\raggedright\arraybackslash}X" * (n - 1) + "}"
                    end_ = r"\end{tabularx}"
                else:
                    begin, end_ = r"\begin{tabular}{l" + "r" * (n - 1) + "}", r"\end{tabular}"
                L.append(r"\begin{table}[H]\centering\small")
                if caption:
                    L.append(r"{\textbf{" + latex_inline(caption) + r"}}\\[3pt]")
                L += [begin, r"\toprule",
                      " & ".join(r"\textbf{" + latex_inline(h) + "}" for h in header) + r" \\", r"\midrule"]
                L += [" & ".join(latex_inline(c) for c in r) + r" \\" for r in rows]
                L += [r"\bottomrule", end_, r"\end{table}"]
            elif kind == "figure":
                rel = Path(os.path.relpath(b[1], folder)).as_posix()
                L += [r"\begin{figure}[H]\centering",
                      rf"\includegraphics[width={b[3]}\textwidth]{{{rel}}}",
                      r"\par\small\textit{" + latex_inline(b[2]) + "}", r"\end{figure}"]
        L.append(r"\end{document}")
        return "\n".join(L)

    # ------------------------------------------------------------------ output
    def save(self, folder, stem="report") -> str:
        """Write <stem>.md and <stem>.tex into folder and try to compile <stem>.pdf.
        Returns a one-line status message."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{stem}.md").write_text(self.to_markdown(folder), encoding="utf-8")
        (folder / f"{stem}.tex").write_text(self.to_latex(folder), encoding="utf-8")
        if not shutil.which("pdflatex"):
            return f"wrote {stem}.md and {stem}.tex (pdflatex not found, so no PDF)"
        try:
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{stem}.tex"],
                               cwd=folder, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            return f"wrote {stem}.md and {stem}.tex (pdflatex timed out, so no PDF)"
        for ext in (".aux", ".log", ".out"):
            if r.returncode == 0:
                (folder / f"{stem}{ext}").unlink(missing_ok=True)
        if r.returncode != 0:
            return f"wrote {stem}.md and {stem}.tex (pdflatex failed, see {stem}.log)"
        return f"wrote {stem}.md, {stem}.tex and {stem}.pdf"


def fmt(v, nd=0):
    """Number with thousands separators, e.g. fmt(5704.6) -> '5,705'."""
    return f"{v:,.{nd}f}"


def json_default(o):
    """Lets json.dump write NumPy numbers and arrays."""
    import numpy as np
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"cannot write {type(o)} to JSON")
