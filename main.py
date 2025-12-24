from flask import Flask, render_template, url_for
import json

app = Flask(__name__)


@app.context_processor
def inject_config():
    with open("configs/info_panel.json", encoding="utf-8") as f:
        info_bar = json.load(f)

    return {
        'info_panel': info_bar,
    }

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)