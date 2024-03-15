import numpy as np
import requests
from flask import Flask, request
from inference import output_keypoints_from_fn

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    """Basic HTML response."""
    body = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h1>Hello World</h1>"
        "</body>"
        "</html>"
    )
    print(type(body))
    return body


def save_img_from_url_to_fn(url, fn):
    r = requests.get(url)
    with open(fn,'wb') as f: 
        f.write(r.content)


@app.route("/predict", methods=["POST"])
def predict():
    data_json = request.get_json()
    save_fn = "./test.png"
    img_url = data_json["img_url"]

    save_img_from_url_to_fn(img_url, save_fn)
    pred = output_keypoints_from_fn(save_fn)
    return str(pred)

if __name__ == "__main__":
    app.run()