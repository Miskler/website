from flask import Flask, render_template, url_for
import json

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
    return render_template('experience.html', experience=exp, experience_per_day="0.28")

if __name__ == '__main__':
    app.run(debug=True)