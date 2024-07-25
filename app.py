import joblib
from flask import Flask, render_template, redirect, url_for, request
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import math
from DiseaseModel import DiseaseModel
from helper import prepare_symptoms_array
heart_dis = pickle.load(open('model\heart_disease_model.pkl', 'rb'))
diab_dis = pickle.load(open('model\diabetes_model.pkl', 'rb'))
park_dis = pickle.load(open('model\parkinsons_model.pkl', 'rb'))

app = Flask(__name__)

disease_model = DiseaseModel()
disease_model.load_xgboost('model/xgboost_model.json')
x = disease_model.all_symptoms
y = x.to_list()

@app.route('/')
def dashboard():
    return render_template("dashboard.html")

@app.route('/symptoms')
def disease():
    return render_template("disease.html", symptoms=y)

@app.route("/disindex")
def disindex():
    return render_template("disindex.html")


@app.route("/parkinson")
def cancer():
    return render_template("parkinson.html")


@app.route("/diabetes")
def diabetes():
    return render_template("diabetes.html")


@app.route("/heart")
def heart():
    return render_template("heart.html")

@app.route("/liver")
def liver():
    return render_template("liver.html")

@app.route("/predict", methods = ["POST"])
def predict():
    symptoms = request.form.getlist('symptoms')
    #print(symptoms)
    X = prepare_symptoms_array(symptoms)
    prediction, prob = disease_model.predict(X)

    precautions = disease_model.predicted_disease_precautions()
    prec = ""
    for i in range(4):
        prec += precautions[i] + "\n"
    #formatted_precautions = [f'{i+1}. {precaution}' for i, precaution in enumerate(precautions)]
    print(prec)
    descr = disease_model.describe_predicted_disease()
    #print(type(formatted_precautions))
    return render_template('result_dis.html', disease=prediction, probability=round(prob*100,2), desc= descr, precaution = prec)

def ValuePredictor2(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1, size)
    if (size == 22):
        loaded_model = joblib.load(r"model\parkinsons_model.pkl")
        result = loaded_model.predict(to_predict)
    return result[0]

@app.route("/predictpark", methods = ["POST"])
def predictpark():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        #to_predict_list["Total_Protiens"] = math.log(to_predict_list["Total_Protiens"])
        #print( to_predict_list)
        if (len(to_predict_list) == 22):
            result = ValuePredictor2(to_predict_list, 22)
            
        if(int(result)==1):
            prediction = "Sorry, you have chances of getting the disease. Please consult the doctor immediately"
        elif(int(result) == 0):
            prediction = "No need to fear! You have no dangerous symptoms of the disease."
        return(render_template("parkinson_result.html", prediction_text=prediction))
    


def ValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1, size)
    if (size == 9):
        loaded_model = joblib.load(r"model\liver.sav")
        result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/predictliver', methods=["POST"])
def predictliver():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        key_to_log = ['Total_Bilirubin','Alkaline_Phosphotase','Alamine_Aminotransferase','Aspartate_Aminotransferase']
        processed_input = {}
        for key, value in to_predict_list.items():
            float_value = float(value)
            if key in key_to_log:
                processed_input[key] = math.log(float_value)
            else:
                processed_input[key] = float_value
        to_predict_list = list(processed_input.values())
        #to_predict_list = list(map(float, to_predict_list))
        #to_predict_list["Total_Protiens"] = math.log(to_predict_list["Total_Protiens"])
        print( to_predict_list)
        if (len(to_predict_list) == 9):
            result = ValuePredictor(to_predict_list, 9)
            
        if(int(result)==1):
            prediction = "Sorry, you have chances of getting the disease. Please consult the doctor immediately"
        elif(int(result) == 0):
            prediction = "No need to fear! You have no dangerous symptoms of the disease."
        return(render_template("result.html", prediction_text=prediction))    







##################################################################################


#####################################################################


@app.route('/predictt', methods=['POST'])
def predictt():
    if request.method == 'POST':
        preg = request.form['pregnancies']
        glucose = request.form['glucose']
        bp = request.form['bloodpressure']
        st = request.form['skinthickness']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        dpf = request.form['dpf']
        age = request.form['age']

        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = diab_dis.predict(data)

        return render_template('diab_result.html', prediction=my_prediction)


############################################################################################################

@app.route('/predictheart', methods=['POST'])
def predictheart():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

    features_name = ["age", "sex", "cp", "trestbps", "chol", "fbs",
                     "restecg", "thalach", "exang", "oldpeak", "slope", "ca",
                     "thal"]

    df = pd.DataFrame(features_value, columns=features_name)
    output = heart_dis.predict(df)

    if output == 1:
        res_val = "a high risk of Heart Disease"
    else:
        res_val = "a low risk of Heart Disease"

    return render_template('heart_result.html', prediction_text='Patient has {}'.format(res_val))


############################################################################################################

if __name__ == "__main__":
    app.run(debug=True)

