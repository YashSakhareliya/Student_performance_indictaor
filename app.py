import sys
from flask import Flask,redirect,render_template,request
from src.exception import CustomException
from src.logger import logging
from src.pipeline.predict_pipeline import Predict_pipeline,CustomObject

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            customdata = CustomObject(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('race_ethnicity'),
                parental_level_of_education=request.form.get('parental_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation'),
                writing_score=float(request.form.get('writing_score')),
                reading_score=float(request.form.get('reading_score')),
            )

            datafream = customdata.get_data_as_data_fream()
            print(datafream)
            
            pred_pipeline = Predict_pipeline()
            pred = pred_pipeline.predict(datafream)
            return render_template('home.html',result=pred)
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)