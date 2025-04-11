# IMPORTING LIBRARIES
import numpy as np
from flask import Flask,request, render_template
import pandas as pd
import pickle

model = pickle.load(open('C:\\Users\\milan\\PycharmProjects\\PythonProject\\PythonProject\\breast_cancer_detection\\cancer_detection.pkl', 'rb'))
# FLASK APP
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    featurs = request.form['feature']
    featurs_list = featurs.split(',')
    np_features = np.asarray(featurs_list, dtype=np.float32)
    pred = model.predict(np_features.reshape(1,-1))
    output = ['cancerous' if pred[0]==1 else 'Not cancerous']
    return render_template('index.html',message=output)


#PYTHON MAIN
if __name__=='__main__':
    app.run(debug=True)