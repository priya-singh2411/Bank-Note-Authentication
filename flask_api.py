# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 19:41:18 2020

@author: priya Singh
"""
#import libraries
import pickle
from flask import Flask, request
#import flasgger
from flasgger import Swagger
#import numpy as np
import pandas as pd

app=Flask(__name__)
Swagger(app)

#Unpickling the classifier
pickle_in=open('classifier.pkl','rb')
classifier=pickle.load(pickle_in)

#@app.route('/')
@app.route('/apidocs')
def welcome_page():
    return '<h1> WELCOME TO THE HOMEPAGE </h1>'

@app.route('/predict',methods=["GET"])
def predict_note_authentication():
    
    """ Let's Authenticate the Bank Note 
    This is using docstring for specification.
    ---
    parameters:
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values         
    
    """
    variance=request.args.get('variance')
    skewness=request.args.get('skewness')
    curtosis=request.args.get('curtosis')
    entropy=request.args.get('entropy')
    predict=classifier.predict([[variance,skewness,curtosis,entropy]])
    if predict[0]==0:
      return 'Predicted value is : ' +str(predict) +' Note is not authenticate'
    else:
        return 'Predicted value is : ' +str(predict) +' Note is Authenticate'

@app.route('/predict_file',methods=["POST"])
def predict_note_file():
    
    """ Let's Authenticate the Bank Note 
    This is using docstring for specification.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
    responses:
        200:
            description: The output values         
    
    """
    df_test=pd.read_csv(request.files.get("file"))
    print(df_test.head())
    prediction=classifier.predict(df_test)
    return str(list(prediction))
    
      


if __name__=='__main__':
    app.run(host='0.0.0.0',port=8000)

#url to run the app : http://127.0.0.1:5000/apidocs/

