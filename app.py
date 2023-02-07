from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle
import sklearn

#Create flask app
app = Flask(__name__)

#Load the model
model = pickle.load(open('C:/Users/Chinh/DS Project/Bodyfat/app/model.pkl', 'rb'))

@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    density = model.predict(features)
    prediction = (490/density)- 450

    return render_template('index.html', prediction_text = 'Your estimated body fat percentage: {}'.format(prediction))

if __name__ == '__main__':
    app.run(debug=True)
