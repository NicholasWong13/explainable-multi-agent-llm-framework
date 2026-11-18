"""Rebuild PROPOSAL_DRAFT.html from PROPOSAL_DRAFT.md so the two never disagree.
    python build_proposal_html.py
Handles the subset of Markdown the proposal uses: headings, paragraphs, tables, bullet and numbered lists,
horizontal rules, **bold**, *italic* and `code`. Standard library only."""
import html
import re
from pathlib import Path

STYLE = """<style>
  body { font-family: Calibri, Arial, sans-serif; font-size: 11pt; line-height: 1.6;
         margin: 2.5cm 2.5cm 2.5cm 3cm; color: #000; }
  h1 { font-size: 16pt; font-weight: bold; text-align: center; margin-bottom: 4pt; }
  h2 { font-size: 13pt; font-weight: bold; margin-top: 18pt; margin-bottom: 4pt;
       border-bottom: 1px solid #555; padding-bottom: 2pt; }
  h3 { font-size: 11pt; font-weight: bold; margin-top: 12pt; margin-bottom: 2pt; }
  p  { margin: 6pt 0; text-align: justify; }
  table { border-collapse: collapse; width: 100%; margin: 10pt 0; font-size: 10pt; }
  th { background: #2E4057; color: #fff; padding: 5pt 8pt; text-align: left; }
  td { border: 1px solid #ccc; padding: 4pt 8pt; vertical-align: top; }
  tr:nth-child(even) td { background: #f5f7fa; }
  ul, ol { margin: 4pt 0 4pt 20pt; }
  li { margin-bottom: 3pt; }
  hr { border: none; border-top: 1px solid #ddd; margin: 12pt 0; }
  code { font-family: Consolas, monospace; font-size: 10pt; }
</style>"""


def inline(t):
    t = html.escape(t, quote=False)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", t)
    return t


def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def convert(md):
    out, lines, i = [], md.splitlines(), 0
    title = "Draft proposal"
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if not s:
            i += 1
        elif s == "---":
            out.append("<hr>")
            i += 1
        elif s.startswith("#"):
            lvl = len(s) - len(s.lstrip("#"))
            text = s[lvl:].strip()
            if lvl == 1:
                title = text
            out.append(f"<h{lvl}>{inline(text)}</h{lvl}>")
            i += 1
        elif s.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(cells(lines[i]))
                i += 1
            head, body = rows[0], [r for r in rows[1:] if not all(re.fullmatch(r":?-+:?", c) for c in r)]
            out.append("<table><tr>" + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr>")
            out += ["<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body]
            out.append("</table>")
        elif re.match(r"^(- |\d+\. )", s):
            tag = "ol" if s[0].isdigit() else "ul"
            items = []
            while i < len(lines) and lines[i].strip():
                t = lines[i].strip()
                if re.match(r"^(- |\d+\. )", t):
                    items.append(re.sub(r"^(- |\d+\. )", "", t))
                else:  # continuation line
                    items[-1] += " " + t
                i += 1
            out.append(f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in items) + f"</{tag}>")
        else:
            para = []
            while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||- |\d+\. |---)", lines[i].strip()):
                para.append(lines[i].strip())
                i += 1
            out.append(f"<p>{inline(' '.join(para))}</p>")
    return (f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n<title>{html.escape(title)}</title>\n'
            f"{STYLE}\n</head>\n<body>\n" + "\n".join(out) + "\n</body>\n</html>\n")


if __name__ == "__main__":
    src = Path(__file__).with_name("PROPOSAL_DRAFT.md")
    dst = src.with_suffix(".html")
    dst.write_text(convert(src.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"wrote {dst}")
