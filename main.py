from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Привет, мир!'

if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=6080)