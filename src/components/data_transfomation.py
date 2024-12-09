import os
import sys
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformconfig:
    preprocessor_obj_path: str = os.path.join('artifact', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.preprocessor_obj_config = DataTransformconfig()
    
    def get_data_transformation_obj(self):
        try:
            num_feature = ['reading score', 'writing score']
            cat_feature = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ohe',OneHotEncoder())
                ]
            )
            logging.info('Numeric columns encoding complete')

            logging.info('category columns encoding complete')

            preprocessor = ColumnTransformer(
                [
                    ('Num pipeline',num_pipeline,num_feature),
                    ('cat pipeline',cat_pipeline,cat_feature)
                ]
            )
            logging.info("Object ready to be return")

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_set_path, test_set_path):
        try:
            train_df = pd.read_csv(train_set_path)
            test_df = pd.read_csv(test_set_path)
            logging.info("Train and test Data read successfully")

            preprocessor_obj = self.get_data_transformation_obj()
            logging.info("preprocessor obj initiated")
            target_column = 'math score'

            input_train_feature_df = train_df.drop([target_column],axis=1)
            target_train_feature_df = train_df[target_column]

            input_test_feature_df = test_df.drop([target_column],axis=1)
            target_test_feature_df = test_df[target_column]

            logging.info("Data transformation started")

            input_train_feature_transformed = preprocessor_obj.fit_transform(input_train_feature_df)
            input_test_feature_transformed = preprocessor_obj.transform(input_test_feature_df)

            logging.info("Data transformation completed")

            train_arr = np.c_[
                input_train_feature_transformed,np.array(target_train_feature_df)
            ]

            test_arr = np.c_[
                input_test_feature_transformed,np.array(target_test_feature_df)
            ]

            save_object(
                file_path = self.preprocessor_obj_config.preprocessor_obj_path,
                obj = preprocessor_obj
            )

            logging.info("saved preprocessing object:{}".format(self.preprocessor_obj_config.preprocessor_obj_path))

            return(
                train_arr,
                test_arr,
                self.preprocessor_obj_config.preprocessor_obj_path
            )
        except Exception as e:
            raise CustomException(e, sys)
            

