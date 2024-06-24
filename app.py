from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the ridge regressor and standard scaler pickle models
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            Temperature = float(request.form['Temperature'])
            RH = float(request.form['RH'])
            Ws = float(request.form['Ws'])
            Rain = float(request.form['Rain'])
            FFMC = float(request.form['FFMC'])
            DMC = float(request.form['DMC'])
            ISI = float(request.form['ISI'])
            Classes = float(request.form['Classes'])
            Region = float(request.form['Region'])

            # Scale the new data
            new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
            # Make prediction
            result = ridge_model.predict(new_data_scaled)
            return render_template('index.html', result=result[0])
        except Exception as e:
            return render_template('index.html', result=str(e))
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
