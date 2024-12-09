import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_model

# all algorithms
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

@dataclass
class ModelTrainerconfig:
    model_obj_path:str=os.path.join('artifact', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_obj_path_config = ModelTrainerconfig()
    
    def initiate_model_trainer(self,train_data_arr,test_data_arr):
        try:
            X_train,X_test,y_train,y_test = (
                train_data_arr[:,:-1],
                test_data_arr[:,:-1],
                train_data_arr[:,-1],
                test_data_arr[:,-1]
            )
            logging.info("compleate the seperation of data")

            models = {
                'Linear regression':LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                'ElasticNet':ElasticNet(),
                'K-Nearest regressor':KNeighborsRegressor(),
                'DecisionTreeRegressor':DecisionTreeRegressor(),
                'GaussianNB':GaussianNB(),
                'Support Vector Regressor':SVR(),
                'RandomForestRegressor':RandomForestRegressor(),
                'AdaBoostRegressor':AdaBoostRegressor(),
                'GradientBoostingRegressor':GradientBoostingRegressor(),
                'XGBRegressor':XGBRegressor()
            }

            model_report:dict = evaluate_model(X_train,X_test,y_train,y_test,models)

            logging.info("compleate the model evaluation")

            # finding best model score
            best_model_score = max(sorted(model_report.values()))

            # finding best model name
            best_model_name  = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]


            if best_model_score < 0.60:
                raise CustomException("No best Model Found")
            logging.info("Best Model found on both traning and testing dataset")

            # save model
            save_object(
                file_path=self.model_obj_path_config.model_obj_path,
                obj=best_model
            )

            logging.info("Model saved successfully")

            # make prediction
            best_model.fit(X_train, y_train)
            prediction = best_model.predict(X_test)
            logging.info("Model prediction completed successfully")

            # calculate the score
            model_score = r2_score(y_test, prediction)
            logging.info("Model score: {:.4f}".format(model_score))

            return model_score


            
        except Exception as e:
            CustomException(e,sys)
