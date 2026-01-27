import os
from pathlib import Path
from typing import List, Union
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from flask import url_for
from markupsafe import Markup, escape
from PIL import Image

STATIC_ROOT = Path("static/timeline/res")


def wrap_images(html: str, static_root: str = "") -> str:
    soup = BeautifulSoup(html, "html.parser")

    for p in soup.find_all("p"):
        imgs = p.find_all("img", recursive=False)
        if not imgs:
            continue

        for img in imgs:
            src = img.get("src")
            if not src:
                continue

            # Приводим src к строке, если это необходимо
            src_str = str(src)
            parsed = urlparse(src_str)
            if parsed.scheme in ("http", "https"):
                continue

            file_path = os.path.join(static_root, src_str.lstrip("/"))
            if not os.path.isfile(file_path):
                continue

            try:
                with Image.open(file_path) as im:
                    w, h = im.size
            except Exception:
                continue

            # Явно указываем типы атрибутов
            a = soup.new_tag(
                "a",
                href=src_str,
                attrs={
                    "data-pswp-width": str(w),
                    "data-pswp-height": str(h),
                },
            )

            img.wrap(a)

        # Правильная обработка классов
        current_classes = p.get("class")
        if current_classes is None:
            p["class"] = "pswp-gallery"
        else:
            # Преобразуем в строку, если это AttributeValueList
            if not isinstance(current_classes, str):
                current_classes = " ".join(str(c) for c in current_classes)

            classes = current_classes.split()
            if "pswp-gallery" not in classes:
                classes.append("pswp-gallery")
                p["class"] = " ".join(classes)

    return str(soup)


def render_pswp_description(desc: List[Union[str, List[str]]]) -> Markup:
    parts: List[str] = []
    has_gallery = False

    for item in desc:
        # обычный текст
        if isinstance(item, str):
            parts.append(str(escape(item)))
            continue

        # ["text", "path"]
        text, rel_path = item
        img_path = STATIC_ROOT / rel_path

        if not img_path.exists():
            parts.append(str(escape(text)))
            continue

        with Image.open(img_path) as img:
            w, h = img.size

        url = url_for("static", filename=f"timeline/res/{rel_path}")

        parts.append(
            f"<a href='{url}' data-pswp-width='{w}' data-pswp-height='{h}'>" f"{escape(text)}</a>"
        )
        has_gallery = True

    if has_gallery:
        return Markup(f"<div class='pswp-gallery'>{''.join(parts)}</div>")

    return Markup("".join(parts))
