import os
import sys
import pandas as pd
from flask.cli import pass_script_info
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class Predict_pipeline:
    def __init__(self):
        pass

    def predict(self, data_frame):
        try:
            # Load model from file
            preprocessor = load_object('artifact/preprocessor.pkl')
            model = load_object('artifact/model.pkl')
           
            data_scaled = preprocessor.transform(data_frame)
            pred = model.predict(data_scaled)
            return pred
            
        except CustomException as e:
            raise CustomException(e,sys)

class CustomObject:
    def __init__(self,
                 gender,
                 race_ethnicity,
                 parental_level_of_education,
                 lunch,
                 test_preparation_course,
                 reading_score,
                 writing_score):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
    
    def get_data_as_data_fream(self):
        try:
            data_fream = {
            'gender': [self.gender],
            'race/ethnicity': [self.race_ethnicity],
            'parental level of education': [self.parental_level_of_education],
            'lunch': [self.lunch],
            'test preparation course': [self.test_preparation_course],
            'reading score': [self.reading_score],
            'writing score': [self.writing_score]
            }
            return pd.DataFrame(data_fream)
        except CustomException as e:
            raise CustomException(e,sys)

                