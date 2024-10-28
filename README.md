# Algerian Forest Fires Prediction Web Application
This project is a web-based application developed using Flask to predict the likelihood of forest fires based on weather conditions in two Algerian regions, Bejaia and Sidi Bel-abbes.
The dataset contains meteorological data and fire weather indices for these regions, enabling a machine learning model to classify whether conditions are likely to result in a fire or not.
#### Project Structure 

<img width="344" alt="image" src="https://github.com/user-attachments/assets/308138f8-6df5-4350-a14b-f806498f6537">

##### Dataset
The dataset contains 244 observations collected from two regions in Algeria between June and September 2012. The target variable is binary:
"Fire" or "Not Fire." Each observation includes attributes such as temperature, relative humidity, wind speed, and fire weather indices (FWI) components.

**Features**
1. Temperature: Maximum temperature at noon in °C
2. Relative Humidity (RH): Percentage
3. Wind Speed (Ws): Speed in km/h
4. Rain: Total rainfall in mm
5. Fine Fuel Moisture Code (FFMC): Fire Weather Index component
6. Duff Moisture Code (DMC): Fire Weather Index component
7. Drought Code (DC): Fire Weather Index component
8. Initial Spread Index (ISI): Fire Weather Index component
9. Buildup Index (BUI): Fire Weather Index component
10. Fire Weather Index (FWI): Composite index
11. Classes: Binary target variable (Fire/Not Fire)
12. Region: Binary encoding of the regions (Bejaia: 0, Sidi Bel-abbes: 1)
# Model
The prediction model uses Ridge Regression, trained on scaled data, to predict the likelihood of a fire event. 
A StandardScaler is applied to ensure consistency of input data.
**Prerequisites**
pip install flask numpy pickle-mixin

**Running the Application**
git clone "https://github.com/TVsony/END-TO-END-ML-/blob/master/app.py"
python app.py

**Usage**
1.Fill in the input form with the required weather conditions.
2.Submit the form to view the prediction result, which will indicate whether the conditions are likely to result in a fire or not.

**Prediction**
![image](https://github.com/user-attachments/assets/6ceefe58-1fbd-48a1-a5f1-a45ef9726109)


#### File Descriptions
app.py: Main Flask application that handles data inputs, scaling, prediction, and result rendering.
models/ridge.pkl: Pretrained Ridge Regression model used for predictions.
models/scaler.pkl: StandardScaler object to normalize input features.

##### Future Improvements
1. Expand model to handle other machine learning algorithms.
2. Include data visualization for trends and data insights.
3. Add historical fire data for additional model accuracy.




