from flask import Flask, render_template, url_for
import json
from pswp import render_pswp_description, render_md
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

    return render_template('experience.html', experience=exp, experience_per_day="0.55")

if __name__ == '__main__':
    app.run(debug=True)