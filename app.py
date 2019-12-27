# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:56:28 2019

@author: Sabitha Jaleel
"""

import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

app = Flask(__name__)
salary_deploy = pickle.load(open('startup_prediction.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])

def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = salary_deploy.predict(final_features)
    
    output = round(prediction[0],2)
    
    return render_template('index.html',prediction_text ='Profit for the startup should be $ {}'.format(output))

if __name__ == '__main__':
    app.run(debug=True)