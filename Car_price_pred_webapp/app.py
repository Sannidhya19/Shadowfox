from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("rf1.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        present_price = float(request.form['present_price'])
        kms_driven = float(request.form['kms_driven'])
        owner = float(request.form['owner'])
        age = float(request.form['age'])

        fuel = request.form['fuel_type']
        seller = request.form['seller_type']
        transmission = request.form['transmission']

        # One-hot encoding
        fuel_diesel = 1 if fuel == "Diesel" else 0
        fuel_petrol = 1 if fuel == "Petrol" else 0

        seller_individual = 1 if seller == "Individual" else 0
        transmission_manual = 1 if transmission == "Manual" else 0

        features = [[
            present_price,
            kms_driven,
            owner,
            age,
            fuel_diesel,
            fuel_petrol,
            seller_individual,
            transmission_manual
        ]]

        prediction = model.predict(features)
        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text=f"Estimated Price: ₹ {output:,.2f} Lakhs")

    except Exception as e:
        return render_template('index.html', prediction_text="Error: " + str(e))

if __name__ == "__main__":
    app.run(debug=True)