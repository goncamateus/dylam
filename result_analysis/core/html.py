"""Minimal semantic-HTML table assembly shared by the scope table Generators.

Additive sibling to the LaTeX rendering each Generator already had: same
computed values, a second markup target. Emits only bare <table>/<thead>/
<tbody>/<tr>/<th>/<td> elements plus <strong>/<caption> where the LaTeX side
uses \\textbf/a table title -- no style attributes, no hardcoded colors or
fonts, so the Beyond-PDF kit's own CSS is free to style it.

`detex` is not a general LaTeX parser: it strips exactly the small set of
math/text macros the four table Generators' row labels and headers use
(percent signs, pm/times/greek letters, --  as an en dash, \\boldsymbol/
\\textbf/\\emph/\\text wrappers, and bare $...$ math delimiters), so a label
already computed for the LaTeX table renders as readable plain text instead
of literal TeX source.
"""
import re
from html import escape

_MACROS = [
    (r"\\%", "%"),
    (r"\\pm", "\u00b1"),
    (r"\\times", "\u00d7"),
    (r"\\geq", "\u2265"),
    (r"\\leq", "\u2264"),
    (r"\\zeta", "\u03b6"),
    (r"\\ell_1", "\u21131"),
    (r"\\ell", "\u2113"),
    (r"\\mathrm\{e\}", "e"),
    (r"\\log_\{10\}", "log10"),
    (r"\\boldsymbol\{([^}]*)\}", r"\1"),
    (r"\\textbf\{([^}]*)\}", r"\1"),
    (r"\\emph\{([^}]*)\}", r"\1"),
    (r"\\text\{([^}]*)\}", r"\1"),
    (r"--", "\u2013"),
]


def detex(s):
    """Plain-text rendering of the LaTeX math/text macros used in this repo's
    table row labels and headers (see module docstring for the exact set)."""
    out = s
    for pattern, repl in _MACROS:
        out = re.sub(pattern, repl, out)
    out = out.replace("$", "")
    return re.sub(r"\s+", " ", out).strip()


def table(headers, rows, caption=None):
    """Assemble one semantic HTML <table> from plain-text headers/cells.

    headers: list[str], header row (rendered as <th>).
    rows: list[list[str]], one list of cell strings per row (rendered as
        <td>). Wrap a cell in `strong(...)` for semantic emphasis instead of
        hand-writing markup -- see `strong`.
    caption: optional plain-text <caption>.
    """
    parts = ["<table>"]
    if caption:
        parts.append(f"<caption>{escape(caption)}</caption>")
    parts.append("<thead><tr>" + "".join(f"<th>{escape(h)}</th>" for h in headers)
                 + "</tr></thead>")
    parts.append("<tbody>")
    for row in rows:
        cells = "".join(f"<td>{_cell(c)}</td>" for c in row)
        parts.append(f"<tr>{cells}</tr>")
    parts.append("</tbody>")
    parts.append("</table>")
    return "\n".join(parts) + "\n"


def strong(text):
    """Mark one cell's text for semantic emphasis (<strong>), mirroring the
    LaTeX side's \\textbf/\\mathbf on the "best" value in a row. Not inline
    styling -- <strong> is a semantic element, not a style attribute."""
    return _Strong(text)


class _Strong(str):
    """A str subclass so plain cells and strong cells share one code path in
    `table`, distinguished only by isinstance at render time."""


def _cell(c):
    if isinstance(c, _Strong):
        return f"<strong>{escape(str(c))}</strong>"
    return escape(str(c))
