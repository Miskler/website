from PIL import Image
from pathlib import Path
from flask import url_for
from markupsafe import Markup, escape
import markdown

STATIC_ROOT = Path("static/timeline/res")

def render_md(text):
    html = markdown.markdown(
        text,
        extensions=[
            "extra",
            "sane_lists",
            "nl2br"
        ]
    )
    return Markup(html)

def render_pswp_description(desc):
    parts = []
    has_gallery = False

    for item in desc:
        # обычный текст
        if isinstance(item, str):
            parts.append(escape(item))
            continue

        # ["text", "path"]
        text, rel_path = item
        img_path = STATIC_ROOT / rel_path

        if not img_path.exists():
            parts.append(escape(text))
            continue

        with Image.open(img_path) as img:
            w, h = img.size

        url = url_for("static", filename=f"timeline/res/{rel_path}")

        parts.append(
            f"<a href='{url}' data-pswp-width='{w}' data-pswp-height='{h}'>"
            f"{escape(text)}</a>"
        )
        has_gallery = True

    if has_gallery:
        return Markup(f"<div class='pswp-gallery'>{''.join(parts)}</div>")

    return Markup("".join(parts))
