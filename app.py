#!/usr/bin/env python
# coding: utf-8


from flask import Flask, render_template, request
import model.model as mdl
from flask_ngrok import run_with_ngrok

app = Flask(__name__)
run_with_ngrok(app)

my_model = mdl.Model()




@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods= ['POST'])
def predict():
    msg = request.form['mood_pred']
    result = my_model.predict_results(msg)[0]

    if result > 0.5 :
        T = "\"Depressed\" kindly be in touch with us or check specialist"
    else :
        T = "\"Not Depressed\" Keep on That we hope best for you"
    return render_template('index.html',pred = "You are {}".format(T))




app.run()




