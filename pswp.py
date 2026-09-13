import os
from pathlib import Path
from typing import List, Union
from urllib.parse import unquote, urlparse

from bs4 import BeautifulSoup
from flask import url_for
from markupsafe import Markup, escape
from PIL import Image

STATIC_ROOT = Path("static/timeline/res")
AUDIO_MIME_TYPES = {
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
}
VIDEO_MIME_TYPES = {
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
}


def _media_label(href: str, fallback: str) -> str:
    path = unquote(urlparse(href).path)
    stem = Path(path).stem.replace("_", " ").replace("-", " ")
    return " ".join(stem.split()).strip() or fallback


def _audio_label(href: str) -> str:
    return _media_label(href, "Аудиозапись")


def _video_label(href: str) -> str:
    return _media_label(href, "Видеозапись")


def _audio_player(soup: BeautifulSoup, href: str, label: str):
    parsed = urlparse(href)
    mime_type = AUDIO_MIME_TYPES.get(Path(parsed.path).suffix.lower(), "audio/*")

    wrapper = soup.new_tag("span", attrs={"class": "audio-player", "role": "group"})
    caption = soup.new_tag("span", attrs={"class": "audio-player__label"})
    caption.string = label
    wrapper.append(caption)

    audio = soup.new_tag("audio", attrs={"controls": "", "preload": "none", "aria-label": label})
    source = soup.new_tag("source", attrs={"src": href, "type": mime_type})
    audio.append(source)
    wrapper.append(audio)
    return wrapper


def wrap_audio(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Keep support for hand-written audio tags, while adding a visible caption.
    for audio in soup.find_all("audio"):
        if audio.find_parent(class_="audio-player"):
            continue

        source = audio.find("source", src=True)
        if source is None and audio.get("src"):
            source = soup.new_tag("source", attrs={"src": str(audio["src"])})
            audio.append(source)
        if source is None:
            continue

        label = str(audio.get("data-label") or audio.get("aria-label") or "").strip()
        if not label:
            label = _audio_label(str(source["src"]))

        for child in list(audio.contents):
            if child is not source and getattr(child, "name", None) != "source":
                child.extract()

        caption = soup.new_tag("span", attrs={"class": "audio-player__label"})
        caption.string = label
        audio.wrap(soup.new_tag("span", attrs={"class": "audio-player", "role": "group"}))
        audio.parent.insert(0, caption)
        audio["aria-label"] = label

    # Turn Markdown links to audio files into the same player component.
    for link in list(soup.find_all("a", href=True)):
        href = str(link["href"])
        if link.find_parent("audio"):
            continue

        suffix = Path(urlparse(href).path).suffix.lower()
        if suffix not in AUDIO_MIME_TYPES:
            continue

        label = " ".join(link.stripped_strings).strip() or _audio_label(href)
        link.replace_with(_audio_player(soup, href, label))

    return str(soup)


def _video_player(soup: BeautifulSoup, href: str, label: str):
    parsed = urlparse(href)
    mime_type = VIDEO_MIME_TYPES.get(Path(parsed.path).suffix.lower(), "video/*")

    wrapper = soup.new_tag("figure", attrs={"class": "video-player", "role": "group"})
    video = soup.new_tag(
        "video",
        attrs={
            "controls": "",
            "preload": "metadata",
            "playsinline": "",
            "aria-label": label,
        },
    )
    source = soup.new_tag("source", attrs={"src": href, "type": mime_type})
    video.append(source)
    wrapper.append(video)

    caption = soup.new_tag("figcaption", attrs={"class": "video-player__label"})
    caption.string = label
    wrapper.append(caption)
    return wrapper


def wrap_video(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Keep support for hand-written video tags, while adding a visible caption.
    for video in soup.find_all("video"):
        if video.find_parent(class_="video-player"):
            continue

        source = video.find("source", src=True)
        if source is None and video.get("src"):
            source = soup.new_tag("source", attrs={"src": str(video["src"])})
            video.append(source)
        if source is None:
            continue

        label = str(video.get("data-label") or video.get("aria-label") or "").strip()
        if not label:
            label = _video_label(str(source["src"]))

        for child in list(video.contents):
            if child is not source and getattr(child, "name", None) != "source":
                child.extract()

        video["controls"] = ""
        video["preload"] = video.get("preload") or "metadata"
        video["playsinline"] = ""
        video["aria-label"] = label
        wrapper = soup.new_tag("figure", attrs={"class": "video-player", "role": "group"})
        video.wrap(wrapper)
        caption = soup.new_tag("figcaption", attrs={"class": "video-player__label"})
        caption.string = label
        video.parent.append(caption)

    # Turn Markdown links to video files into an inline preview player.
    for link in list(soup.find_all("a", href=True)):
        href = str(link["href"])
        if link.find_parent("video"):
            continue

        suffix = Path(urlparse(href).path).suffix.lower()
        if suffix not in VIDEO_MIME_TYPES:
            continue

        label = " ".join(link.stripped_strings).strip() or _video_label(href)
        link.replace_with(_video_player(soup, href, label))

    return str(soup)


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
