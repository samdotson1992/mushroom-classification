import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('resources/model.pkl', 'rb'))
encoder = pickle.load(open("resources/encoder.pkl" ,'rb') ) 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    data = pd.DataFrame.from_records([ request.form.to_dict() ])

    mushroom_profile =  encoder.transform(data[['cap-surface', 'gill-size', 'spore-print-color', 'population', 'habitat']]).toarray()

    pred = model.predict( mushroom_profile  )

    text = ""

    if pred[0] == 1:
        text = "This mushroom is probably poisonous."

    if pred[0] == 0:
        text = "This mushroom is probably edible"

    return render_template('index.html', prediction_text=text)

if __name__ == "__main__":
    app.run(debug=True)