from flask import Flask, render_template, request
from flaskwebgui import FlaskUI
from pycaret.regression import *
import pandas as pd
import warnings 
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    Temperature = float(request.form.get('Temperature'))
    Humidity = float(request.form.get('Humidity'))
    wind_speed = float(request.form.get('wind_speed'))
    general_diffusion_flow = float(request.form.get('general_diffusion_flow'))
    diffuse_flow = float(request.form.get('diffuse_flow'))
    air_quality_index = int(request.form.get('air_quality_index'))
    cloudiness = int(request.form.get('cloudiness'))
    df_sample = pd.DataFrame({'Temperature': [Temperature], 'Humidity': [Humidity], 'Wind_Speed': [wind_speed], 
                              'general_diffuse_flows': [general_diffusion_flow], 'diffuse_flows': [diffuse_flow],
                              'Air_Quality_Index': [air_quality_index], 'Cloudiness': [cloudiness]})
    data_pred = predict_model(model, df_sample)
    output = round(float(data_pred.iloc[0]['prediction_label']), 2)
    # output = round(float(model.predict(df_sample)[0]), 2)
    return render_template('predict.html', output= output)

if __name__  ==  '__main__':
    model = load_model('rf_model')
    FlaskUI(app=app, server='flask').run()
    # app.run(debug=True)

