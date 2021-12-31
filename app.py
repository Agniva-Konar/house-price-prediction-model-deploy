import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
out= pickle.load(open('LIN_REG.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = out.predict(final_features)
    prediction=round(prediction[0],2)
    return render_template('index.html', prediction_text='The Price of the house should be ${}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)