from flask import Flask, render_template, url_for, abort
import os
import json
from pswp import render_pswp_description, render_md, wrap_images
from datetime import datetime

app = Flask(__name__)


@app.context_processor
def inject_config():
    with open("configs/info_panel.json", encoding="utf-8") as f:
        info_bar = json.load(f)
    with open("configs/navigation.json", encoding="utf-8") as f:
        navigation = json.load(f)

    return {
        'info_panel': info_bar,
        'navigation': navigation
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/experience')
def experience():
    with open("configs/timeline.json", encoding="utf-8") as f:
        exp = json.load(f)
    
    today = datetime.today().strftime("%d.%m.%Y")
    for e in exp:
        if isinstance(e.get("description"), list):
            e["description"] = render_md(render_pswp_description(e["description"]))
        if "point" not in e["timeline"].keys() and e["timeline"]["to"].upper() == "NOW":
            e["timeline"]["to"] = today
            e["timeline"]["now"] = True

    return render_template('experience.html', experience=exp, experience_per_day="0.65")

@app.route('/papers/<path:slug>')
def papers(slug):
    base_dir = os.path.join(app.root_path, 'static', 'papers')
    md_path = os.path.join(base_dir, f'{slug}/paper.md')

    # защита от ../
    if not os.path.abspath(md_path).startswith(os.path.abspath(base_dir)):
        abort(404)

    if not os.path.isfile(md_path):
        abort(404)

    with open(md_path, encoding='utf-8') as f:
        md = f.read()

    html = wrap_images(render_md(md))

    return render_template(
        'paper.html',
        pape=html,
        page_type='paper',
        paper_slug=slug
    )

if __name__ == '__main__':
    app.run(debug=True)