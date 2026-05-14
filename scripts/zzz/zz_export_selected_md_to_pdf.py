from pathlib import Path

import markdown
from xhtml2pdf import pisa

FILES = [
    Path("/triumvirate/home/alexarol/breast_cancer_analysis/BRANE/scripts/SCRIPT_INDEX.md"),
    Path("/triumvirate/home/alexarol/breast_cancer_analysis/BRANE/results/STAGE_CONNECTIONS_MAP.md"),
    Path("/triumvirate/home/alexarol/breast_cancer_analysis/BRANE/results/RESULTS_INDEX.md"),
]

CSS = """
@page { size: A4; margin: 1.8cm; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 11pt; line-height: 1.45; color: #111; }
h1, h2, h3, h4 { color: #0b3d91; }
code, pre { font-family: Courier, monospace; font-size: 9pt; }
pre { background: #f6f8fa; border: 1px solid #ddd; padding: 8px; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #ccc; padding: 6px; vertical-align: top; }
a { color: #0b57d0; }
"""


def convert_markdown_file(md_path: Path) -> None:
    if not md_path.exists():
        print(f"MISSING: {md_path}")
        return

    pdf_path = md_path.with_suffix(".pdf")
    md_text = md_path.read_text(encoding="utf-8")
    html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "toc"])
    html = (
        "<html><head><meta charset='utf-8'><style>"
        + CSS
        + "</style></head><body>"
        + html_body
        + "</body></html>"
    )

    with pdf_path.open("wb") as output_file:
        result = pisa.CreatePDF(html, dest=output_file)

    if result.err:
        print(f"FAILED: {md_path}")
    else:
        print(f"OK: {pdf_path}")


if __name__ == "__main__":
    for path in FILES:
        convert_markdown_file(path)
