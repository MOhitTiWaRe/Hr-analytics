import os
import pickle

import numpy as np
from flask import Flask, render_template, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
model = pickle.load(open('hr_rf_model.pkl', 'rb'))

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/main', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        int_features = [int(x) for x in request.form.values()]
        int_features[0] = int_features[0] / 100
        if int_features[-1] == 1:
            int_features.append(0)
            int_features.append(0)
        elif int_features[-1] == 2:
            int_features[-1] = 0
            int_features.append(1)
            int_features.append(0)
        elif int_features[-1] == 3:
            int_features[-1] = 0
            int_features.append(0)
            int_features.append(1)
        final = [np.array(int_features)]
        prediction = model.predict(final)
        if prediction == 1:
            result = "Employee is likely to leave the company."
        elif prediction == 0:
            result = "Employee is likely to continue working for company."
        return render_template('main.html', pred=result)
    return render_template('main.html', pred='')

if __name__ == "__main__":
    app.run(debug=True)
