from flask import Flask, render_template, request
import numpy as np
import pickle
from sklearn.preprocessing import RobustScaler

app = Flask(__name__,template_folder='templates')
model = pickle.load(open('GaussianNB_prediction_model1.py', 'rb'))


@app.route('/', methods=['GET'])
def home():
    return render_template('liver_data.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Age = int(request.form['Age'])
        Gender = int(request.form['Gender'])
        Total_Bilirubin = float(request.form['Total_Bilirubin'])
        Direct_bilirubin = float(request.form['Direct_Bilirubin'])
        Alkaline_Phosphotase = float(request.form['Alkaline_Phosphotase'])
        Alamine_Aminotransferase = float(request.form['Alamine_Aminotransferase'])
        Aspartate_Aminotransferase = float(request.form['Aspartate_Aminotransferase'])
        Total_Protiens = float(request.form['Total_Protiens'])
        Albumin = float(request.form['Albumin'])
        Albumin_and_Globulin_Ratio = float(request.form['Albumin_and_Globulin_Ratio'])

        v1 = RobustScaler()
        values1 = v1.fit_transform([[Age, Gender, Total_Bilirubin, Direct_bilirubin, Alkaline_Phosphotase,
                                     Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens, Albumin,
                                     Albumin_and_Globulin_Ratio]])



        prediction = model.predict(values1)

        return render_template('liver_message.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)

