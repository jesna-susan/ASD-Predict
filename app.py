from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        A1 = int(request.form['A1'])
        A2 = int(request.form['A2'])
        A3 = int(request.form['A3'])
        A4 = int(request.form['A4'])
        A5 = int(request.form['A5'])
        A6 = int(request.form['A6'])
        A7 = int(request.form['A7'])
        A8 = int(request.form['A8'])
        A9 = int(request.form['A9'])
        A10 = int(request.form['A10'])
        age = int(request.form['age'])
        jaundice = int(request.form['jaundice'])
        sex = int(request.form['sex']) 
        family_member = int(request.form['family_member'])

        features = np.array([[A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, age, jaundice, sex, family_member]])  
        prediction = model.predict(features)
        result = prediction.tolist()
        return render_template('result.html', result=result)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/survey_form.html')
def survey_form():
    return render_template('survey_form.html')

@app.route('/result.html')
def result():
    return render_template('result.html')


if __name__ == '__main__':
    model = joblib.load(r'D:\ASDPREDICT\logistic_regression_model_test2 (1).pkl')
    app.run(debug=True)

    model.close()
