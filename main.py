# Dependencies
from flask import Flask, request, jsonify, json
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np
from getCUBEMS import loaddata
from getWEATHER import loadtemp
from keras.models import model_from_json, load_model

app = Flask(__name__)

#automatically load the result when the page is loaded
@app.route('/MLR', methods=['GET'])
def MLR():
    # read data from CUBEMS and scrape temperature data
    load = loaddata()
    temp = loadtemp()
    
    # forecast using MLR
    forecastMLR = regressor.predict([[load, temp]])
    
    # forecast using ANN
    df1 = pd.read_csv('minmax.csv', header=None, index_col=0)
    nLoad = (load-df1.loc['Lmin'][1])/(df1.loc['Lmax'][1]-df1.loc['Lmin'][1])
    nTemp = (temp-df1.loc['Tmin'][1])/(df1.loc['Tmax'][1]-df1.loc['Tmin'][1])
    nW = (1-df1.loc['Wmin'][1])/(df1.loc['Wmax'][1]-df1.loc['Wmin'][1])
    nH = (23-df1.loc['Hmin'][1])/(df1.loc['Hmax'][1]-df1.loc['Hmin'][1])
    a = np.array([[nTemp, nLoad, nW, nH]])
    print(a)
    testPredict = modelANN.predict(a)
    print(testPredict)
    forecastANN = testPredict[0][0]*(df1.loc['Lmax'][1]-df1.loc['Lmin'][1])+df1.loc['Lmin'][1] 
    print(forecastANN)
    
    return '''    
        <html>
            <body>
                <p>Lt (from CUBEMS) = {load:.2f} kW </p>
                <p>Tt+1 (from Weather.com) = {temp:.2f} deg C </p>
                <p><h3>Lt+1 Forecasted load (MLR method) = {forecastMLR:.2f} kW </h3></p>
                <p><h3>Lt+1 Forecasted load (ANN method) = {forecastANN:.2f} kW </h3></p>
            </body>
        </html>
    '''.format(load=load, temp=temp, forecastMLR=forecastMLR[0], forecastANN=forecastANN)

# call from postman API
@app.route('/apiMLR', methods=['POST'])
def apiMLR():
    if regressor:
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            print(query)
            #query = query.reindex(columns=model_columns, fill_value=0)
            prediction = list(regressor.predict(query))
            return jsonify({'prediction': str(prediction)})
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')



if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    # Load MLR model
    regressor = joblib.load("modelMLR.pkl") # Load "model.pkl"
    print ('ANN model loaded')

    # Load ANN model 
    modelANN = load_model('modelANN.h5')
    print('ANN model loaded')

    app.run(port=port, debug=True)