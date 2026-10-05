"""
=================================================================
@copyright: A. Ivanovitch | CEO SKYL4R | 2026
=================================================================

Builds an academic-style PDF (preprint layout) from a Markdown paper.

    python3 docs/build_paper.py            # docs/PAPER_V2.md -> docs/paper/skylar2_paper.pdf (+ .html)

Layout: serif body (Linux Libertine), numbered sections as written in the Markdown, abstract as an
indented block, booktabs-style tables, figures from files, running page numbers. No logo, no
coloured boxes.

Math is typeset with matplotlib's mathtext (Computer Modern), so no TeX installation is needed:
  $$ ... $$            display equation, numbered automatically
  $$ ... $$ {#eq:id}   display equation with a label, referenced in the text as [@eq:id]
  $ ... $              inline math
mathtext supports a large LaTeX subset (\\frac, \\sqrt, \\sum, \\mathrm, \\odot, \\top, ...). Unsupported
syntax fails the build instead of printing raw LaTeX.

Front matter: the first lines of the Markdown, before the first `## `, in the form
  # Title
  **Authors:** ...
  **Affiliation:** ...
  **Date:** ...
The section `## Abstract` is rendered as the abstract block.
"""

import argparse
import base64
import html
import io
import os
import re
import sys

import markdown

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSS = """
@page {
  size: A4; margin: 24mm 24mm 22mm 24mm;
  @bottom-center { content: counter(page); font: 9pt/1 'Linux Libertine O', serif; }
}
@page :first { @bottom-center { content: none } }
html { font-size: 10.5pt }
body { font-family: 'Linux Libertine O', 'Liberation Serif', serif; line-height: 1.38;
       color: #000; text-align: justify; hyphens: auto; font-variant-numeric: lining-nums; }
.title { font-size: 17pt; font-weight: bold; text-align: center; line-height: 1.25; margin: 4mm 6mm 5mm;
         hyphens: none; }
.authors { text-align: center; font-size: 11.5pt; margin-bottom: 1mm }
.affil { text-align: center; font-size: 10pt; font-style: italic; margin-bottom: 1mm }
.date { text-align: center; font-size: 9.5pt; margin-bottom: 7mm }
.abstract { margin: 0 11mm 7mm; font-size: 9.6pt; line-height: 1.35 }
.abstract .head { text-align: center; font-weight: bold; font-size: 10pt; margin-bottom: 1.5mm }
h2 { font-size: 12pt; font-weight: bold; margin: 6mm 0 2.5mm; text-align: left; hyphens: none;
     break-after: avoid; }
h3 { font-size: 10.5pt; font-weight: bold; margin: 4.5mm 0 1.8mm; text-align: left; hyphens: none;
     break-after: avoid; }
h4 { font-size: 10.5pt; font-weight: bold; font-style: italic; margin: 3mm 0 1mm; break-after: avoid }
p { margin: 0 0 2.2mm; orphans: 3; widows: 3 }
a { color: #000; text-decoration: none }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: .82em }
pre { font-family: 'DejaVu Sans Mono', monospace; font-size: 7.8pt; line-height: 1.35;
      margin: 2.5mm 4mm; text-align: left; break-inside: avoid; white-space: pre-wrap }
ul, ol { margin: 0 0 2.4mm; padding-left: 6mm }
li { margin-bottom: .8mm }
blockquote { margin: 2.5mm 6mm; font-size: 9.6pt }
table { border-collapse: collapse; margin: 1.5mm auto 4mm; font-size: 8.8pt; break-inside: avoid;
        border-top: 1pt solid #000; border-bottom: 1pt solid #000; }
th { font-weight: bold; border-bottom: .6pt solid #000; padding: 1.1mm 2.4mm; text-align: left;
     vertical-align: bottom }
td { padding: .8mm 2.4mm; text-align: left; vertical-align: top }
td[style*="right"], th[style*="right"] { white-space: nowrap }
.caption { font-size: 9pt; margin: 3mm 3mm 1mm; text-align: justify; break-after: avoid }
.caption b { font-weight: bold }
.fig { text-align: center; margin: 3mm 0 1mm; break-inside: avoid }
.fig img { max-width: 100%; }
.figcap { font-size: 9pt; margin: 1mm 3mm 4mm; text-align: justify }
.eq { display: table; width: 100%; margin: 2.2mm 0; break-inside: avoid }
.eq .body { display: table-cell; text-align: center; vertical-align: middle }
.eq .num { display: table-cell; width: 10mm; text-align: right; vertical-align: middle; font-size: 10pt }
img.im { vertical-align: middle }
.refs p { font-size: 8.8pt; line-height: 1.3; padding-left: 5mm; text-indent: -5mm; margin-bottom: 1.2mm;
          text-align: left; hyphens: none }
hr { border: none; border-top: .5pt solid #000; margin: 5mm 0 }
"""

_EQ_COUNTER = [0]
_EQ_LABELS = {}


def _mathtext_svg(tex, size):
    import matplotlib
    matplotlib.use("svg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["mathtext.fontset"] = "cm"
    matplotlib.rcParams["svg.fonttype"] = "path"
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${tex}$", fontsize=size)
    buf = io.StringIO()
    try:
        fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.012, transparent=True)
    except Exception as e:
        raise SystemExit(f"\nmathtext cannot typeset:\n  {tex}\n  ({e})\n")
    finally:
        plt.close(fig)
    svg = buf.getvalue()
    w = float(re.search(r'width="([\d.]+)pt"', svg).group(1))
    h = float(re.search(r'height="([\d.]+)pt"', svg).group(1))
    return svg, w, h


def _img(svg, w, h, cls, style=""):
    b64 = base64.b64encode(svg.encode()).decode()
    return (f'<img class="{cls}" style="width:{w:.2f}pt;height:{h:.2f}pt;{style}" '
            f'src="data:image/svg+xml;base64,{b64}">')


def convert_math(md):
    def display(m):
        tex, label = m.group(1).strip(), m.group(2)
        _EQ_COUNTER[0] += 1
        n = _EQ_COUNTER[0]
        if label:
            _EQ_LABELS[label] = n
        svg, w, h = _mathtext_svg(tex, 11.5)
        return (f'\n\n<div class="eq"><span class="body">{_img(svg, w, h, "dm")}</span>'
                f'<span class="num">({n})</span></div>\n\n')

    md = re.sub(r"\$\$(.+?)\$\$[ \t]*(?:\{#(eq:[\w-]+)\})?", display, md, flags=re.S)

    def inline(m):
        svg, w, h = _mathtext_svg(m.group(1), 10.5)
        return _img(svg, w, h, "im", "vertical-align:-0.25em")

    # inline $...$ (not inside code spans): protect code spans first
    codes = []

    def keep(m):
        codes.append(m.group(0))
        return f"\x00{len(codes) - 1}\x00"

    md = re.sub(r"`[^`\n]+`", keep, md)
    md = re.sub(r"(?<![\\$])\$([^$\n]+?)\$", inline, md)
    md = re.sub(r"\x00(\d+)\x00", lambda m: codes[int(m.group(1))], md)
    md = re.sub(r"\[@(eq:[\w-]+)\]", lambda m: f"({_EQ_LABELS.get(m.group(1), '??')})", md)
    return md


def _inline_md(text):
    """Markdown inline (corsivo, grassetto, codice) dentro una didascalia HTML."""
    html_ = markdown.markdown(text)
    return re.sub(r"^<p>(.*)</p>$", r"\1", html_.strip(), flags=re.S)


def _check_inline_math(md):
    """Una formula inline non deve andare a capo nel sorgente: altrimenti gli $ si
    accoppiano male e il resto del paragrafo diventa matematica."""
    bad = []
    for n, line in enumerate(md.splitlines(), 1):
        stripped = re.sub(r"`[^`]*`", "", line).replace("$$", "")
        if stripped.count("$") % 2:
            bad.append(f"  riga {n}: {line.strip()[:90]}")
    if bad:
        raise SystemExit("Formula inline spezzata su due righe (numero dispari di $):\n" + "\n".join(bad))


def figures(md, src_dir):
    """![caption](path) on its own line -> centred figure with numbered caption."""
    n = [0]

    def fig(m):
        n[0] += 1
        cap, path = m.group(1), m.group(2)
        full = path if os.path.isabs(path) else os.path.join(src_dir, path)
        return (f'\n\n<div class="fig"><img src="file://{full}"></div>'
                f'<div class="figcap"><b>Figure {n[0]}.</b> {_inline_md(cap)}</div>\n\n')

    return re.sub(r"^!\[(.+?)\]\((.+?)\)\s*$", fig, md, flags=re.M)


def table_captions(md):
    """A line `Table: caption` immediately before a table -> numbered caption above it."""
    n = [0]

    def cap(m):
        n[0] += 1
        return f'\n<div class="caption"><b>Table {n[0]}.</b> {_inline_md(m.group(1))}</div>\n'

    return re.sub(r"^Table:\s*(.+)$", cap, md, flags=re.M)


def build(src, out):
    text = open(src, encoding="utf-8").read()
    head, _, body_md = text.partition("\n## ")
    body_md = "## " + body_md
    title = re.search(r"^#\s+(.+)$", head, re.M).group(1).strip()
    field = lambda k: (re.search(rf"^\*\*{k}:\*\*\s*(.+)$", head, re.M) or [None, ""])[1].strip()

    abstract = ""
    m = re.match(r"## Abstract\s*\n(.+?)(?=\n## )", body_md, re.S)
    if m:
        abstract = m.group(1).strip()
        body_md = body_md[m.end():]

    # references: one entry per paragraph, whatever the spacing in the source
    if "## References" in body_md:
        pre, refs = body_md.split("## References", 1)
        refs = re.sub(r"\n(?=\[\d+\] )", "\n\n", refs)
        body_md = pre + "## References" + refs

    src_dir = os.path.dirname(os.path.abspath(src))
    _check_inline_math(re.sub(r"\$\$.+?\$\$", "", body_md, flags=re.S))
    body_md = table_captions(figures(convert_math(body_md), src_dir))
    abstract_html = markdown.markdown(convert_math(abstract))
    ext = ["tables", "fenced_code", "attr_list", "sane_lists"]
    body_html = markdown.markdown(body_md, extensions=ext)
    # references: one paragraph per entry (hanging indent), even if written as one block with line breaks
    body_html = re.sub(r'(<h2[^>]*>References</h2>)(.*)$',
                       lambda m: m.group(1) + '<div class="refs">'
                       + re.sub(r'<br\s*/?>\s*', '</p><p>', m.group(2)) + '</div>', body_html, flags=re.S)

    doc = f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>
<div class="title">{html.escape(title)}</div>
<div class="authors">{html.escape(field('Authors'))}</div>
<div class="affil">{html.escape(field('Affiliation'))}</div>
<div class="date">{html.escape(field('Date'))}</div>
<div class="abstract"><div class="head">Abstract</div>{abstract_html}</div>
{body_html}
</body></html>"""
    os.makedirs(os.path.dirname(out), exist_ok=True)
    html_out = out[:-4] + ".html"
    open(html_out, "w", encoding="utf-8").write(doc)
    from weasyprint import HTML
    HTML(string=doc, base_url=src_dir).write_pdf(out)
    return html_out, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(REPO, "docs", "PAPER_V2.md"))
    ap.add_argument("--out", default=os.path.join(REPO, "docs", "paper", "skylar2_paper.pdf"))
    a = ap.parse_args()
    h, p = build(a.src, a.out)
    print(f"HTML -> {h}\nPDF  -> {p} ({os.path.getsize(p) / 1024:.0f} KB)")
