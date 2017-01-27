import json
import numpy as np
from flask import Flask, send_from_directory

app = Flask(__name__, static_url_path='')

data = {}

def set_data(field, value):
    print "{} were set".format(field)
    if field == "front_view":
        value_range = np.max(value) - np.min(value)
        if value_range != 0:
            value = (value - np.min(value)) / value_range * 255.
        value = np.clip(value, 0, 255).astype(np.uint8)

    data[field] = value

@app.route('/')
def index():
    return send_from_directory('', "index.html")

@app.route('/images/<path:path>')
def image(path):
    return send_from_directory('../../data/', path)

@app.route("/js/<path:path>")
def send_js(path):
    return send_from_directory('js', path)

@app.route("/data/<field_name>")
def getdata(field_name):
    return json.dumps(data[field_name].tolist())

def start():
    app.run()
