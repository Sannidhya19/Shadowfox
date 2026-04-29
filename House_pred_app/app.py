from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("rf.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        RM = float(request.form["RM"])
        LSTAT = float(request.form["LSTAT"])
        PTRATIO = float(request.form["PTRATIO"])
        INDUS = float(request.form["INDUS"])
        TAX = float(request.form["TAX"])
        NOX = float(request.form["NOX"])

        features = np.array([[RM, LSTAT, PTRATIO, INDUS, TAX, NOX]])

        prediction = model.predict(features)

        return render_template(
            "index.html",
            prediction_text=f"Predicted House Price: ${round(prediction[0]*1000):,}"
        )

    except:
        return render_template("index.html", prediction_text="Invalid Input!")

if __name__ == "__main__":
    app.run(debug=True)
