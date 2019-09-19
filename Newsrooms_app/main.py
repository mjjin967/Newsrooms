from flask import Flask
import json

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/hello-json')
def hello_json():
    article1 = {"title": "hello1", "detail": "detail"}
    article2 = {"title": "hello2", "detail": "detail"}
    article3 = {"title": "hello3", "detail": "detail"}
    return json.dumps([article1, article2, article3])

if __name__ == '__main__':
    app.run()