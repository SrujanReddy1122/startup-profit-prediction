import flask
from flask import Flask,render_template,request
import numpy as np
import pickle
app = Flask(__name__)
from sklearn.linear_model import LinearRegression

with open("MLR_MODEL (1).pkl","rb") as f:
    model = pickle.load(f)

app = Flask(__name__)
@app.route("/")
def fun():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def prediction():

    rnd = float(request.form["rnd"])
    admin = float(request.form["admin"])
    marketing = float(request.form["marketing"])
    state = float(request.form["state"])


    features = np.array([[rnd, admin, marketing, state]])


    prediction_result = model.predict(features)[0]


    formatted_result = f"{prediction_result:,.2f}"

    return render_template("index.html", prediction_text=formatted_result)


if __name__ == "__main__":
    app.run(debug=True)