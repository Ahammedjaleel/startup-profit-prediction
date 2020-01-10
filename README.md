# startup_prediction
Startup Prediction
ML-Model-Flask-Deployment
Prerequisites
You must have Scikit Learn, Pandas (for Machine Leraning Model) and Flask (for API) installed.

Project Structure
This project has four major parts :

startup_profit_prediction.py - This contains code fot our Machine Learning model to predict employee salaries absed on trainign data in 'hiring.csv' file.
app.py - This contains Flask APIs that receives details through GUI or API calls, computes the precited value based on our model and returns it.
request.py - This uses requests module to call APIs already defined in app.py and dispalys the returned value.
templates - This folder contains the HTML template to allow user to enter employee detail and displays the predicted employee salary.
Running the project
This would create a serialized version of our model into a file startup_prediction.pkl
Run app.py using below command to start Flask API
python app.py
By default, flask will run on port 5000.

Navigate to URL http://localhost:5000
You should be able to view the homepage 

Enter valid numerical values in all 4 input boxes and hit Predict.

If everything goes well, you should be able to see the predcited salary vaule on the HTML page

You can also send direct POST requests to FLask API using Python's inbuilt request module 
