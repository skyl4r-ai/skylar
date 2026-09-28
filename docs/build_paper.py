"""
=================================================================
@copyright: A. Ivanovitch | skyl4r.ai | 2026
=================================================================

Costruisce il PDF del technical report da `docs/PAPER_V2.md`.

    python docs/build_paper.py                     # -> docs/paper/skylar2_paper.pdf
    python docs/build_paper.py --src docs/PAPER.md --out docs/paper/skylar1.pdf

Perché uno script e non un comando: le formule. WeasyPrint non fa MathML né LaTeX,
e in un paper le equazioni rese male si notano prima del contenuto. Le display
equation sono quindi tradotte a mano in HTML+unicode (dizionario `EQUATIONS`) e
quelle inline da un traduttore di pattern. Se si aggiunge un'equazione al markdown
e non la si registra qui, il build **fallisce con un errore esplicito** invece di
stampare LaTeX grezzo dentro il PDF.
"""

import argparse
import html
import os
import re
import sys

import markdown

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Display equation: LaTeX sorgente → HTML reso a mano ──────────────────────
# La chiave è una sottostringa distintiva dell'equazione nel markdown.
EQUATIONS = {
    "S_t = ": (
        '<span class="eq"><i>S</i><sub>t</sub> = '
        '( <i>I</i> − β<sub>t</sub> <i>k</i><sub>t</sub> <i>k</i><sub>t</sub><sup>⊤</sup> ) '
        '· Diag(α<sub>t</sub>) · <i>S</i><sub>t−1</sub> '
        '+ β<sub>t</sub> <i>k</i><sub>t</sub> <i>v</i><sub>t</sub><sup>⊤</sup></span>'
    ),
    "H_{kda}": (
        '<span class="eq boxed"><i>H</i><sub>kda</sub> = '
        '<span class="frac"><span class="num">n<sub>heads</sub> + n<sub>kv&nbsp;heads</sub></span>'
        '<span class="den">2</span></span></span>'
    ),
    "k_l = ": (
        '<span class="eq"><i>k</i><sub>l</sub> = RMSNorm(<i>v</i><sub>l</sub>)'
        '<span class="sp"></span>'
        '<i>p</i> = softmax<sub>l</sub>( <i>k</i><sub>l</sub> · <i>q</i> )'
        '<span class="sp"></span>'
        '<i>o</i> = Σ<sub>l</sub> <i>p</i><sub>l</sub> <i>v</i><sub>l</sub></span>'
    ),
    "\\text{score}_l": (
        '<span class="eq">score<sub>l</sub> = '
        '<span class="frac"><span class="num"><i>v</i><sub>l</sub> · <i>wq</i></span>'
        '<span class="den">rms(<i>v</i><sub>l</sub>)</span></span>'
        '<span class="sp"></span><i>wq</i> = <i>w</i> ⊙ <i>q</i> '
        '<span class="note">(precalcolato una volta)</span></span>'
    ),
    "\\text{situ}(x)": (
        '<span class="eq">situ(<i>x</i>) = '
        'β<sub>1</sub> tanh( <i>W</i><sub>1</sub><i>x</i> / β<sub>1</sub> ) ⊙ '
        'σ( <i>W</i><sub>1</sub><i>x</i> ) ⊙ '
        'β<sub>2</sub> tanh( <i>W</i><sub>3</sub><i>x</i> / β<sub>2</sub> )</span>'
    ),
    "y = W_o": (
        '<span class="eq"><i>y</i> = <i>W</i><sub>o</sub> '
        '[ σ(<i>W</i><sub>g</sub> <i>x</i>) ⊙ RMSNorm(<i>õ</i>) ]</span>'
    ),
    "\\mathrm{LR}(N, D)": (
        '<span class="eq">LR(<i>N</i>, <i>D</i>) = 3×10<sup>−4</sup> · '
        '( <i>N</i> / 0.98×10<sup>9</sup> )<sup>−0.2219</sup> · '
        '( <i>D</i> / 20.37×10<sup>9</sup> )<sup>−0.3509</sup></span>'
    ),
}

# ── Math inline: pattern → sostituzione ─────────────────────────────────────
INLINE = [
    (r"\$2\\,d\\,d_h\\,\(H \+ kv\)\$", "2·<i>d</i>·<i>d</i><sub>h</sub>·(H + kv)"),
    (r"\$4\\,d\\,d_h\\,H_\{kda\}\$", "4·<i>d</i>·<i>d</i><sub>h</sub>·<i>H</i><sub>kda</sub>"),
    (r"\$B\{\\cdot\}T = 65\{,\}536\$", "B·T = 65 536"),
    (r"\$B\{\\cdot\}T = 8192\$", "B·T = 8192"),
    (r"\$B\{\\cdot\}T\$", "B·T"),
    (r"\$\[B\{\\cdot\}T, D\]\$", "[B·T, D]"),
    (r"\$\\beta_1\\beta_2 = 100\$", "β₁β₂ = 100"),
    (r"\$\\beta_1\{=\}4\$", "β₁ = 4"),
    (r"\$\\beta_2\{=\}25\$", "β₂ = 25"),
    (r"\$\\sigma\{=\}0\.05\$", "σ = 0.05"),
    (r"\$2d\$", "2<i>d</i>"),
    (r"\$D\$", "<i>D</i>"),
    (r"\$d\^2\$", "<i>d</i>²"),
    (r"\$d\$", "<i>d</i>"),
    (r"\$L\$", "<i>L</i>"),
    (r"\$H \\cdot d_h = d_\\text\{model\}\$", "H·d<sub>h</sub> = d<sub>model</sub>"),
    (r"\$d_\{ff\}/d_\\text\{model\} = 3\.50\$", "d<sub>ff</sub>/d<sub>model</sub> = 3.50"),
    (r"\$d_\{ff\} = 7168 = 56 \\times 128\$", "d<sub>ff</sub> = 7168 = 56×128"),
    (r"\$L/d\$", "L/d"),
    (r"\$\\mathcal\{N\}\(0, 0\.02/\\sqrt\{2L\}\)\$", "𝒩(0, 0.02/√(2L))"),
    (r"\$\\tanh\(z\) \\approx z\$", "tanh(z) ≈ z"),
    (r"\$1/\\ln 2\$", "1/ln 2"),
    (r"\$n\$", "<i>n</i>"),
    (r"\$t\$", "<i>t</i>"),
    (r"\$S\{=\}4\{-\}6\$", "S = 4–6"),
    (r"\$\\mathrm\{RMSNorm\}\(v\)\\cdot\(w \\odot q\) = \(v \\cdot wq\)\\,/\\,\\mathrm\{rms\}\(v\)\$",
     "RMSNorm(v)·(w⊙q) = (v·wq)/rms(v)"),
]

CSS = """
@page {
  size: A4; margin: 21mm 19mm 20mm 19mm;
  @bottom-center { content: counter(page); font: 8.5pt/1 'DejaVu Serif', serif; color: #8a8a8a; }
  @top-right { content: "Skylar 2 — technical report"; font: 7.5pt/1 'DejaVu Sans', sans-serif;
               color: #b4b4b4; letter-spacing: .04em; }
}
@page :first { @top-right { content: none } @bottom-center { content: none } }

html { font-size: 10pt }
body { font-family: 'DejaVu Serif', Georgia, serif; line-height: 1.52; color: #17181a;
       text-align: justify; hyphens: auto; }

h1 { font-family: 'DejaVu Sans', sans-serif; font-size: 20pt; line-height: 1.22; font-weight: 700;
     letter-spacing: -.015em; margin: 0 0 6mm; text-align: left; hyphens: none; color: #0b0c0e; }
h2 { font-family: 'DejaVu Sans', sans-serif; font-size: 12.5pt; font-weight: 700; margin: 9mm 0 3mm;
     padding-bottom: 1.6mm; border-bottom: .6pt solid #d8dade; text-align: left; hyphens: none;
     break-after: avoid; }
h3 { font-family: 'DejaVu Sans', sans-serif; font-size: 10.5pt; font-weight: 700; margin: 6mm 0 2mm;
     text-align: left; hyphens: none; break-after: avoid; color: #26282c; }
p { margin: 0 0 2.6mm }

a { color: #1a4f9c; text-decoration: none }
strong { font-weight: 700; color: #000 }
em { font-style: italic }

code, tt { font-family: 'DejaVu Sans Mono', monospace; font-size: .855em;
           background: #f2f3f5; padding: .5pt 1.6pt; border-radius: 2px; }
pre { font-family: 'DejaVu Sans Mono', monospace; font-size: 8pt; line-height: 1.42;
      background: #f7f8fa; border: .5pt solid #e2e4e8; border-left: 2pt solid #b8bcc4;
      padding: 2.6mm 3mm; margin: 3mm 0; overflow-x: auto; text-align: left;
      break-inside: avoid; border-radius: 2px; }
pre code { background: none; padding: 0; font-size: 1em }

table { width: 100%; border-collapse: collapse; margin: 3mm 0 4mm; font-size: 8.4pt;
        break-inside: avoid; font-family: 'DejaVu Sans', sans-serif; }
th { font-weight: 700; text-align: left; border-bottom: .9pt solid #2b2d31; padding: 1.5mm 2mm;
     background: #fafbfc; }
td { border-bottom: .4pt solid #e6e8ec; padding: 1.3mm 2mm; vertical-align: top }
tr:last-child td { border-bottom: .9pt solid #2b2d31 }
td:not(:first-child), th:not(:first-child) { text-align: right }
td:first-child, th:first-child { text-align: left }

blockquote { margin: 3.5mm 0; padding: 2.8mm 3.5mm; background: #f6f8fb;
             border-left: 2.2pt solid #7d97bd; font-size: 9pt; break-inside: avoid;
             border-radius: 0 2px 2px 0; }
blockquote p:last-child { margin-bottom: 0 }

ul, ol { margin: 0 0 3mm; padding-left: 5.5mm }
li { margin-bottom: 1.4mm }

hr { border: none; border-top: .4pt solid #dfe1e5; margin: 6mm 0 }

.eq { display: block; text-align: center; margin: 4mm auto; font-size: 10.5pt;
      font-family: 'DejaVu Serif', serif; break-inside: avoid; }
.eq.boxed { border: .7pt solid #2b2d31; padding: 2.2mm 5mm; display: table;
            margin: 4.5mm auto; border-radius: 2px; background: #fcfcfd; }
.eq .sp { display: inline-block; width: 9mm }
.eq .note { font-size: 8.5pt; color: #6a6d73 }
.frac { display: inline-block; vertical-align: middle; text-align: center; margin: 0 1.5mm }
.frac .num { display: block; padding: 0 1.5mm; border-bottom: .6pt solid #17181a }
.frac .den { display: block; padding: 0 1.5mm }

/* frontespizio */
.cover { text-align: center; margin: 0 0 10mm }
.cover .logo { width: 26mm; margin: 6mm auto 8mm; display: block }
.cover .title { font-family: 'DejaVu Sans', sans-serif; font-size: 21pt; font-weight: 700;
                line-height: 1.2; letter-spacing: -.02em; margin: 0 auto 5mm; max-width: 150mm;
                text-align: center; hyphens: none; }
.cover .authors { font-family: 'DejaVu Sans', sans-serif; font-size: 10.5pt; margin-bottom: 1.5mm }
.cover .affil { font-size: 9pt; color: #55585e; margin-bottom: 5mm }
.cover .meta { font-family: 'DejaVu Sans Mono', monospace; font-size: 8pt; color: #7a7d83;
               letter-spacing: .02em }
.cover .rule { width: 30mm; border-top: 1pt solid #17181a; margin: 7mm auto }
/* referenze: rientro sporgente, come si usa */
h2#references + p { font-size: 8.6pt; line-height: 1.42; text-align: left; hyphens: none }
h2#references + p br { line-height: 2.6 }

h2#abstract { border: none; text-align: center; font-size: 10.5pt; letter-spacing: .1em;
              text-transform: uppercase; margin-top: 0 }
"""


def convert_math(md: str) -> str:
    """Sostituisce display e inline math con HTML reso a mano."""
    # display: $$...$$ su una o più righe
    def _display(m):
        body = m.group(1)
        for key, repl in EQUATIONS.items():
            if key in body:
                return repl
        raise SystemExit(
            f"\nEquazione non registrata in EQUATIONS di docs/build_paper.py:\n\n  {body.strip()}\n\n"
            "Aggiungila (con la sua resa HTML) invece di lasciare che il PDF stampi LaTeX grezzo."
        )
    md = re.sub(r"\$\$(.+?)\$\$", _display, md, flags=re.S)
    for pat, repl in INLINE:
        md = re.sub(pat, repl, md)
    if "$" in re.sub(r"`[^`]*`", "", md):
        leftover = [l for l in md.splitlines() if "$" in l and "`" not in l]
        print(f"⚠ math inline non convertita su {len(leftover)} righe:", file=sys.stderr)
        for l in leftover[:5]:
            print("   ", l.strip()[:100], file=sys.stderr)
    return md


def build(src, out, title_lines=None):
    md_text = open(src, encoding="utf-8").read()

    # Il frontespizio si costruisce dalle prime righe e viene rimosso dal corpo.
    lines = md_text.splitlines()
    title = lines[0].lstrip("# ").strip()
    rest = "\n".join(lines[1:])
    # rimuove le due righe di intestazione autore/data che diventano il frontespizio
    rest = re.sub(r"^\s*\*\*A\. Ivanovitch\*\*.*?\n.*?\n", "", rest, count=1, flags=re.S)
    rest = rest.lstrip("\n-").lstrip()

    body_md = convert_math(rest)
    body = markdown.markdown(body_md, extensions=["tables", "fenced_code", "attr_list", "sane_lists"])
    body = body.replace("<h2>Abstract</h2>", '<h2 id="abstract">Abstract</h2>')
    body = body.replace("<h2>References</h2>", '<h2 id="references">References</h2>')

    logo = os.path.join(REPO, "docs", "brand", "skylar-logo-light-2048.png")
    logo_tag = f'<img class="logo" src="file://{logo}">' if os.path.exists(logo) else ""

    cover = f"""<div class="cover">
      {logo_tag}
      <div class="title">{html.escape(title)}</div>
      <div class="authors">A. Ivanovitch</div>
      <div class="affil">skyl4r.ai</div>
      <div class="rule"></div>
      <div class="meta">Technical report · 31 July 2026<br>github.com/skyl4r-ai/skylar</div>
    </div>"""

    doc = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head>" \
          f"<body>{cover}{body}</body></html>"

    os.makedirs(os.path.dirname(out), exist_ok=True)
    html_out = out.replace(".pdf", ".html")
    open(html_out, "w", encoding="utf-8").write(doc)

    from weasyprint import HTML
    HTML(string=doc, base_url=REPO).write_pdf(out)
    return html_out, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(REPO, "docs", "PAPER_V2.md"))
    ap.add_argument("--out", default=os.path.join(REPO, "docs", "paper", "skylar2_paper.pdf"))
    a = ap.parse_args()
    h, p = build(a.src, a.out)
    print(f"HTML → {h}\nPDF  → {p}  ({os.path.getsize(p)/1024:.0f} KB)")
