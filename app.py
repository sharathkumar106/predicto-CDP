# Dependencies
from flask import Flask, request, jsonify, render_template, make_response
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np
import requests
import pytemperature as temp
from datetime import datetime
import statistics as stat



# Your API definition
app = Flask(__name__)


def dataForecast():
    api_key = '4252e170392144144bd44128bf483b39'
    api_id = '5f379804714b52bf40e0e57f'
    api_call = 'http://api.agromonitoring.com/agro/1.0/soil?polyid='+ api_id +'&appid=' + api_key
    weather_api_call = 'http://api.agromonitoring.com/agro/1.0/weather/forecast?polyid='+ api_id +'&appid=' + api_key


    temperature = list()
    humidity =list()
    wind = list()
    soil_moisture = list()
    soil_temp = list()


    json_data_soil = requests.get(api_call).json()
    soil_sample_date = datetime.utcfromtimestamp((json_data_soil["dt"])).strftime('%d %B %Y')
    soil_moisture.append(json_data_soil["moisture"]*100)
    soil_temp.append(float("{:.2f}".format(temp.k2c(json_data_soil["t0"]))))


    json_data_weather = requests.get(weather_api_call).json()


    for item in json_data_weather:

        # Time of the weather data received, partitioned into 3 hour blocks
        weather_sample_date_end = datetime.utcfromtimestamp((item["dt"])).strftime("%d-%m-%Y %H:%M:%S")


        # Split the time into date and hour [2018-04-15 06:00:00]
        temperature.append(float("{:.2f}".format(temp.k2c(item['main']['temp']))))
        humidity.append(float("{:.2f}".format(item["main"]["humidity"])))
        wind.append(item["wind"]["speed"])


        weather = {
        "Temperature": stat.mean(temperature),
        "Humidity":stat.mean(humidity),
        "Wind Speed":stat.mean(wind),
        "Soil Moisture": stat.mean(soil_moisture),
        "Soil Temperature": stat.mean(soil_temp)
        }
        return weather


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/inputs')
def inputs():
    data = dataForecast()
    return render_template('inputs.html', data = data)

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/predict', methods=['POST'])
def predict():
    if rfc:
        try:
            features = [float(x) for x in request.form.values()]
            final_features = [np.array(features)]
            print(final_features)

            prediction = rfc.predict(final_features)

            output = prediction[0]
            print(request.form.values())
            return render_template('result.html',prediction_val='{}'.format(output))

        except:
            print(request.form.values())
            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')


if __name__ == '__main__':
    port = 12345 # The port will be set to 12345
    rfc = joblib.load("model.pkl") # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')

    app.run(port=port, debug=True)
