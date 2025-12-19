import joblib


import numpy as np
import pandas as pd
from flask import Flask, Response, app, render_template, request

application=Flask(__name__)
app=application

# scaler=pickle.load(open('Model/Standardscaler.pkl','rb'))
model = joblib.load('Model/lightgbm.pkl')

# Route for Homepage

@app.route('/')
def index():
    return render_template('home.html')

# Route for Single data point Prediction
@app.route('/',methods=['GET','POST'])
def predict_datapoint():
    result=""
    
    if request.method=='POST':
        Age=int(request.form.get("Age"))
        Gender=float(request.form.get('Gender'))
        Polyuria=float(request.form.get("Polyuria"))
        suddenweightloss=float(request.form.get("Sudden_weight_loss"))
        Polyphagia=float(request.form.get("Polyphagia"))
        visualblurring=float(request.form.get("Visual_blurring"))
        Irritability=float(request.form.get("Irritability"))        
        delayedhealing=float(request.form.get("Delayed_healing"))       
        Polydipsia=float(request.form.get("Polydipsia"))
        Obesity=float(request.form.get("Obesity"))
        
        new_data=([[Age, delayedhealing, suddenweightloss, visualblurring, Obesity, Polyphagia, Polyuria, Gender, Polydipsia, Irritability]])
        predict=model.predict(new_data)
        
        if predict[0]==1:
            result='Diabetic'
        else:
            result='Non-Diabetic'
            
        return render_template("home.html", result=result)
    
    else:
        return render_template('home.html')
    
    
if __name__ == "__main__":
    app.run(host="0.0.0.0")