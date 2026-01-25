import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from flask import Flask, Response, abort, render_template, request
from flask_minify import Minify
from PIL import Image

from github import fetch_github_data
from pswp import render_pswp_description, wrap_images
from steam import get_user_data
from tools import render_md

app = Flask(__name__)

# Или с дополнительными настройками:
Minify(
    app=app,
    html=True,        # минификация HTML
    js=True,          # минификация inline JS
    cssless=True,     # минификация inline CSS
    caching_limit=200,
    static=True,      # минификация статических файлов (ограниченно)
    bypass=[],        # список endpoint’ов, которые не минифицировать
)


with open("configs/secrets.json", encoding="utf-8") as f:
    SECRETS: Dict[str, Any] = json.load(f)


@app.context_processor
def inject_config() -> Dict[str, Any]:
    with open("configs/info_panel.json", encoding="utf-8") as f:
        info_bar: Dict[str, Any] = json.load(f)
    with open("configs/navigation.json", encoding="utf-8") as f:
        navigation: Dict[str, Any] = json.load(f)

    def get_size(path: str, static_root: Path = Path("static")) -> List[int]:
        path = os.path.join(static_root, path)
        if not os.path.isfile(path):
            return [200, 200]

        with Image.open(path) as im:
            w, h = im.size

        min_size = 200
        if w < min_size or h < min_size:
            scale = max(min_size / w, min_size / h)
            w = int(round(w * scale))
            h = int(round(h * scale))

        return [w, h]

    info_bar["avatar"]["size"] = get_size(info_bar["avatar"]["url"])
    for i in info_bar["avatar"]["extra"]:
        i["size"] = get_size(i["url"])

    return {"info_panel": info_bar, "navigation": navigation}


# ------------------------
# Routes
# ------------------------


@app.route("/cards/steam")
async def steam() -> str:
    steam_data = await get_user_data()

    return render_template(
        "cards/steam.html",
        steam_user=steam_data["user"],
        steam_badges=steam_data["badges"],
        steam_games=steam_data["games"],
    )


@app.route("/cards/github")
async def github() -> str:
    result = await fetch_github_data(SECRETS["github"], SECRETS["github_id"])

    return render_template(
        "cards/github.html",
        contributions=result["monthly_contributions"],
        organizations=result["organizations"],
        profile=result["profile"],
        repositories=result["repositories"],
    )


@app.route("/")
async def home() -> str:
    return render_template("index.html")


@app.route("/get/cv")
async def get_cv() -> str:
    return render_template("get_cv.html")


@app.route("/get/cv/ok")
async def get_cv_ok() -> Union[str, Tuple[Response, int]]:
    password = request.args.get("psw")

    if str(password) == str(SECRETS["password_cv"]):
        return render_template("cv.txt")
    else:
        abort(403, description="Неверный пароль")


@app.route("/experience")
async def experience() -> str:
    with open("configs/timeline.json", encoding="utf-8") as f:
        exp: List[Dict[str, Any]] = json.load(f)

    today = datetime.today().strftime("%d.%m.%Y")

    for e in exp:
        if isinstance(e.get("description"), list):
            e["description"] = render_md(render_pswp_description(e["description"]))

        if "point" not in e["timeline"] and e["timeline"]["to"].upper() == "NOW":
            e["timeline"]["to"] = today
            e["timeline"]["now"] = True

    return render_template("experience.html", experience=exp, experience_per_day="0.65")

@app.route("/papers/<path:slug>")
async def papers(slug: str) -> str:
    base_dir = os.path.join(app.root_path, "static", "papers")
    md_path = os.path.join(base_dir, slug, "paper.md")

    if not os.path.abspath(md_path).startswith(os.path.abspath(base_dir)):
        abort(404, description="Directory traversal атака")

    if not os.path.isfile(md_path):
        abort(404, description="Статья не найдена")

    with open(md_path, encoding="utf-8") as f:
        md = f.read()

    html = wrap_images(render_md(md))

    return render_template("paper.html", pape=html, page_type="paper", paper_slug=slug)


# ------------------------
# Errors
# ------------------------


@app.errorhandler(404)
async def error_404(e: Any) -> Tuple[str, int]:
    return (
        render_template(
            "error.html",
            code=404,
            title="Страница не найдена",
            message=e.description
            or ("Данной страницы не существует, " "она удалена или адрес введён неверно"),
        ),
        404,
    )


@app.errorhandler(403)
async def error_403(e: Any) -> Tuple[str, int]:
    return (
        render_template(
            "error.html",
            code=403,
            title="Доступ заблокирован",
            message=e.description or ("Доступ заблокирован, обратитесь к администратору"),
        ),
        403,
    )


@app.errorhandler(500)
async def error_500(e: Any) -> Tuple[str, int]:
    return (
        render_template(
            "error.html",
            code=500,
            title="Внутренняя ошибка сервера",
            message="Произошла ошибка при обработке запроса",
        ),
        500,
    )


if __name__ == "__main__":
    app.run(debug=True)
