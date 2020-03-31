# Dependencies
from flask import Flask, request, jsonify, json
from flask import render_template
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np
from getCUBEMS import loaddata
from getWEATHER import loadtemp
from keras.models import model_from_json, load_model
from datetime import datetime, time

app = Flask(__name__)

@app.route('/chartforecast', methods=['GET'])
def chartforecast():
    df2 = pd.read_csv('output.csv')
    start = df2.iloc[0,0]
    end = df2.iloc[df2.count()[0]-1,0]
    df2['Datetime'] = pd.to_datetime(df2['Datetime'],format="%Y-%m-%d %H:%M:%S")
    df2.set_index('Datetime', inplace=True)
    datetime_index = pd.date_range(start, end, freq='60min')
    data = df2.reindex(datetime_index)
    data['actual'] = data['Lt'].shift(-1)
    data.interpolate(method='linear', inplace=True)
    data.fillna(0, inplace=True)
    legend = 'Actual'
    legend2 = 'Predicted LSTM'
    actual = list(data.actual.values)
    predicted = list(data.Lt1LSTM.values)
    times = list(np.arange(len(actual)))
    return render_template('chart.html', values=actual, values2=predicted, labels=times, legend=legend, legend2=legend2)

 
@app.route('/timechart', methods=['GET'])
def timechart():
    legend = 'Temperatures'
    legend2 = 'Temperatures2'
    temperatures = [73.7, 73.4, 73.8, 72.8, 68.7, 65.2,
                    61.8, 58.7, 58.2, 58.3, 60.5, 65.7,
                    70.2, 71.4, 71.2, 70.9, 71.3, 71.1]
    temperatures2 = [77.7, 75.4, 78, 78, 77, 68,
                    61.8, 58.7, 58.2, 58.3, 60.5, 65.7,
                    70.2, 71.4, 71.2, 70.9, 71.3, 71.1]
    times = [time(hour=11, minute=14, second=15),
             time(hour=11, minute=14, second=30),
             time(hour=11, minute=14, second=45),
             time(hour=11, minute=15, second=00),
             time(hour=11, minute=15, second=15),
             time(hour=11, minute=15, second=30),
             time(hour=11, minute=15, second=45),
             time(hour=11, minute=16, second=00),
             time(hour=11, minute=16, second=15),
             time(hour=11, minute=16, second=30),
             time(hour=11, minute=16, second=45),
             time(hour=11, minute=17, second=00),
             time(hour=11, minute=17, second=15),
             time(hour=11, minute=17, second=30),
             time(hour=11, minute=17, second=45),
             time(hour=11, minute=18, second=00),
             time(hour=11, minute=18, second=15),
             time(hour=11, minute=18, second=30)]
    return render_template('time_chart.html', values=temperatures, values2=temperatures2, labels=times, legend=legend, legend2=legend2)

 
#automatically load the result when the page is loaded
@app.route('/MLR', methods=['GET'])
def MLR():
    now = datetime.now()
    
    #get data from output to df2
    df2 = pd.read_csv('output.csv')
    num_rows = df2.count()[0]

    #if datetime of the last row equals current hour, 
    #do the forecast and write the output
    if (df2.iloc[num_rows-1,0] != now.strftime("%Y-%m-%d %H:00")):
       
        #get load data from CUBEMS and temperature from weather.com
        load = loaddata()
        temp = loadtemp()

        #forecast using MLR
        forecastMLR = regressor.predict([[load, temp]])[0]
        
        #get parameters for denormalization for ANN & LSTM
        df1 = pd.read_csv('minmax.csv', header=None, index_col=0)
        nLoad = (load-df1.loc['Lmin'][1])/(df1.loc['Lmax'][1]-df1.loc['Lmin'][1])
        nTemp = (temp-df1.loc['Tmin'][1])/(df1.loc['Tmax'][1]-df1.loc['Tmin'][1])
        nW = (now.weekday()-df1.loc['Wmin'][1])/(df1.loc['Wmax'][1]-df1.loc['Wmin'][1])
        nH = ((now.hour+1)-df1.loc['Hmin'][1])/(df1.loc['Hmax'][1]-df1.loc['Hmin'][1])
        
        # forecast using ANN
        a = np.array([[nTemp, nLoad, nW, nH]])
        testPredict = modelANN.predict(a)
        forecastANN = testPredict[0][0]*(df1.loc['Lmax'][1]-df1.loc['Lmin'][1])+df1.loc['Lmin'][1] 
        
        #forecast using LSTM
        a = a.reshape((a.shape[0], 1, 4))
        PredictLSTM = modelLSTM.predict(a)
        forecastLSTM = PredictLSTM[0][0]*(df1.loc['Lmax'][1]-df1.loc['Lmin'][1])+df1.loc['Lmin'][1] 
        
        #write to CSV
        import csv   
        fields=[now.strftime("%Y-%m-%d %H:00"), load, temp, forecastMLR, forecastANN, forecastLSTM]
        with open('output.csv', 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(fields)

    else: #if the current hour has already been forecasted, read already forecasted 
          #data from csv file
        load = df2.iloc[num_rows-2,1]
        temp = df2.iloc[num_rows-2,2]
        forecastMLR = df2.iloc[num_rows-2,3]
        forecastANN = df2.iloc[num_rows-2,4]
        forecastLSTM = df2.iloc[num_rows-2,5]

    return '''    
        <html>
            <body>
                <meta http-equiv="refresh" content="3600" > 
                <p><h3>Today is day:{day}, month:{month}, year:{year}</h3></p>
                <p>Current time is {hour}:{minute}</p>
                <p><h3>Load at time t={hour}:00 (from CUBEMS) = {load:.2f} kW </h3></p>
                <p>Forecasted temperature at t={hour1}:00 (from Weather.com) = {temp:.2f} deg C </p>
                <p><h3>Forecasted load at t={hour1}:00 (MLR method) = {forecastMLR:.2f} kW </h3></p>
                <p><h3>Forecasted load at t={hour1}:00 (ANN method) = {forecastANN:.2f} kW </h3></p>
                <p><h3>Forecasted load at t={hour1}:00 (LSTM method) = {forecastLSTM:.2f} kW </h3></p>
            
            
            </body>
        </html>
    '''.format(load=load, temp=temp, forecastMLR=forecastMLR, forecastLSTM=forecastLSTM, forecastANN=forecastANN, day=now.day, month=now.month, year=now.year, hour=now.hour, minute=now.minute, hour1=now.hour+1)

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
    print ('MLR model loaded')

    # Load ANN model 
    modelANN = load_model('modelANN.h5')
    print('ANN model loaded')

    # Load LSTM model 
    modelLSTM = load_model('modelLSTM.h5')
    print('LSTM model loaded')

    
    app.run(port=port, debug=True)