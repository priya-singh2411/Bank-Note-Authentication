# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 20:31:10 2020

@author: pritz
"""

#import libraries
import pickle
#from flask import Flask, request
#import flasgger
#from flasgger import Swagger
#import numpy as np
#import pandas as pd
import streamlit as st
from PIL import Image
#app=Flask(__name__)
#Swagger(app)

#Unpickling the classifier
pickle_in=open('classifier.pkl','rb')
classifier=pickle.load(pickle_in)

#@app.route('/')
#@app.route('/home')
def welcome_page():
    return '<h1> WELCOME TO THE HOMEPAGE </h1>'

#@app.route('/predict',methods=["GET"])
def predict_note_authentication(variance,skewness,curtosis,entropy):
    
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
    # variance=request.args.get('variance')
    # skewness=request.args.get('skewness')
    # curtosis=request.args.get('curtosis')
    # entropy=request.args.get('entropy')
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    print(prediction)
    return prediction


    
def main():
    st.title('Bank Authenticator')
    html_temp= """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Bank Authenticator ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    variance=st.text_input('variance','Type Here')
    skewness=st.text_input('skewness','Type Here')
    curtosis=st.text_input('curtosis','Type Here')
    entropy=st.text_input('entropy','Type Here')
    result=""
    if st.button("Preict"):
        result=predict_note_authentication(variance,skewness,curtosis,entropy)
    st.success('The Output is {}'.format(result()))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()
    
#url to run the app : http://127.0.0.1:5000/apidocs/


