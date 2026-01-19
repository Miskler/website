from PIL import Image
from pathlib import Path
from flask import url_for
from markupsafe import Markup, escape
import markdown
import os
from bs4 import BeautifulSoup
from urllib.parse import urlparse


STATIC_ROOT = Path("static/timeline/res")


def wrap_images(html: str, static_root: str = '') -> str:
    soup = BeautifulSoup(html, "html.parser")

    for p in soup.find_all("p"):
        imgs = p.find_all("img", recursive=False)
        if not imgs:
            continue

        for img in imgs:
            src = img.get("src")
            if not src:
                continue

            parsed = urlparse(src)
            if parsed.scheme in ("http", "https"):
                continue

            file_path = os.path.join(static_root, src.lstrip("/"))
            if not os.path.isfile(file_path):
                continue

            try:
                with Image.open(file_path) as im:
                    w, h = im.size
            except Exception:
                continue

            a = soup.new_tag(
                "a",
                href=src,
                **{
                    "data-pswp-width": str(w),
                    "data-pswp-height": str(h),
                }
            )

            img.wrap(a)

        classes = p.get("class", [])
        if "pswp-gallery" not in classes:
            classes.append("pswp-gallery")
        p["class"] = classes

    return str(soup)


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
