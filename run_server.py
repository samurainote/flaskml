from sklearn.externals import joblib
import flask
from flask import Flask
import numpy as np 

app = Flask(__name__)
model = None

def load_model():
    global model
    model = joblib.load("polynomial_regression.pkl")
    print("load done")

# root with not http, but local file path
@app.route("/predict", methods=["POST"])
def predict():
    # write meta data for retsponse data
    response = {
        "success": False,
        "Content-Type": "application/json"
    }
    # ensure an feature was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.get_json().get("feature"):
            # read feature from json
            feature = flask.request.get_json().get("feature")
            # preprocess for classification
            # list  -> np.ndarray
            feature = np.array(feature).reshape((1, -1))
            # classify the input feature, list because it'S json file
            response["prediction"] = model.predict(feature).tolist()
            # indicate that the request was a success
            response["success"] = True

    return flask.jsonify(response)


if __name__ == "__main__":
    load_model()
    print("flask starting server..")
    app.run()





