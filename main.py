from flask import Flask, render_template, abort, request
import os
from PIL import Image
from pathlib import Path
import json
from steam import get_user_data
from datetime import datetime
from pswp import render_pswp_description, render_md, wrap_images
from pprint import pprint

app = Flask(__name__)


@app.context_processor
def inject_config():
    with open("configs/info_panel.json", encoding="utf-8") as f:
        info_bar = json.load(f)
    with open("configs/navigation.json", encoding="utf-8") as f:
        navigation = json.load(f)

    def get_size(path, static_root: Path = Path("static")) -> list[int, int]:
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

    return {
        "info_panel": info_bar,
        "navigation": navigation
    }


# ------------------------
# Routes
# ------------------------

@app.route("/cards/steam")
async def steam():
    steam_data = await get_user_data()
    pprint(steam_data["user"])

    return render_template(
        "cards/steam.html",
        steam_user=steam_data["user"],
        steam_badges=steam_data["badges"],
        steam_games=steam_data["games"]
    )

@app.route("/")
async def home():
    return render_template("index.html")

@app.route("/get/cv")
async def get_cv():
    return render_template("get_cv.html")

@app.route("/get/cv/ok")
async def get_cv_ok():
    password = request.args.get("psw")
    
    with open("configs/secrets.json", encoding="utf-8") as f:
        SECRETS = json.load(f)

    print(password)
    print(str(SECRETS["password_cv"]))
    if str(password) == str(SECRETS["password_cv"]):
        return render_template("cv.txt")
    else:
        abort(403, description="Неверный пароль")


@app.route("/experience")
async def experience():
    with open("configs/timeline.json", encoding="utf-8") as f:
        exp = json.load(f)

    today = datetime.today().strftime("%d.%m.%Y")

    for e in exp:
        if isinstance(e.get("description"), list):
            e["description"] = render_md(
                render_pswp_description(e["description"])
            )

        if (
            "point" not in e["timeline"]
            and e["timeline"]["to"].upper() == "NOW"
        ):
            e["timeline"]["to"] = today
            e["timeline"]["now"] = True

    return render_template(
        "experience.html",
        experience=exp,
        experience_per_day="0.65"
    )


@app.route("/papers/<path:slug>")
async def papers(slug):
    base_dir = os.path.join(app.root_path, "static", "papers")
    md_path = os.path.join(base_dir, slug, "paper.md")

    if not os.path.abspath(md_path).startswith(os.path.abspath(base_dir)):
        abort(404, description="Directory traversal атака")

    if not os.path.isfile(md_path):
        abort(404, description="Статья не найдена")

    with open(md_path, encoding="utf-8") as f:
        md = f.read()

    html = wrap_images(render_md(md))

    return render_template(
        "paper.html",
        pape=html,
        page_type="paper",
        paper_slug=slug
    )


# ------------------------
# Errors
# ------------------------

@app.errorhandler(404)
async def error_404(e):
    return render_template(
        "error.html",
        code=404,
        title="Страница не найдена",
        message=e.description or (
            "Данной страницы не существует, "
            "она удалена или адрес введён неверно"
        )
    ), 404

@app.errorhandler(403)
async def error_403(e):
    return render_template(
        "error.html",
        code=403,
        title="Доступ заблокирован",
        message=e.description or (
            "Доступ заблокирован, обратитесь к администратору"
        )
    ), 403


@app.errorhandler(500)
async def error_500(e):
    return render_template(
        "error.html",
        code=500,
        title="Внутренняя ошибка сервера",
        message="Произошла ошибка при обработке запроса"
    ), 500


if __name__ == "__main__":
    app.run(debug=True)
