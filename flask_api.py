# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 20:21:05 2020

@author: priya singh
"""
#import libraries
import pickle
from flask import Flask, request
#import numpy as np
import pandas as pd

app=Flask(__name__)

#Unpickling the classifier
pickle_in=open('classifier.pkl','rb')
classifier=pickle.load(pickle_in)

@app.route('/')
@app.route('/home')
def welcome_page():
    return '<h1> WELCOME TO THE HOMEPAGE </h1>'

@app.route('/predict')
def predict_note_authentication():
    variance=request.args.get('variance')
    skewness=request.args.get('skewness')
    curtosis=request.args.get('curtosis')
    entropy=request.args.get('entropy')
    predict=classifier.predict([[variance,skewness,curtosis,entropy]])
    return 'Predicted value is : ' +str(predict)

@app.route('/predict_file',methods=["POST"])
def predict_note_file():
        df_test=pd.read_csv(request.files.get("file"))
        print(df_test.head())
        prediction=classifier.predict(df_test)
        return 'Predicted value for the CSV is : ' +str(list(prediction))
    
      


if __name__=='__main__':
    app.run(debug=True)

