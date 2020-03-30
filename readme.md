this project2 demonstrates:

(a) how we can create API end points to call a machine learning model
    @app.route('/apiMLR', methods=['POST'])

(b) how we can call pre-saved ML model for load forecasting and display data
    @app.route('/MLR', methods=['GET'])

What is next:
* Get time and display time/forecast based on time => DONE
* Consider night & weekend
* Save forecasted data to CSV file (make sure do not save duplicate entry)
* Plot from CSV