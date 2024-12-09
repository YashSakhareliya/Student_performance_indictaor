import os
import sys
from src.exception import CustomException
from src.logger import logging
import dill
from sklearn.metrics import r2_score

def save_object(file_path,obj):
    '''
    This function takes a filename and an object as input and saves the object as a pickle file.
    '''
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj,file_obj)

        logging.info(f"Object saved in: {file_path}")
    except Exception as e:
        logging.error(f"Error saving object in: {file_path} - {str(e)}")
        raise CustomException(e,sys)
    
def evaluate_model(X_train, X_test,y_train,y_test,models):
    '''
    This function evaluates the performance of a given machine learning model report.
    '''
    logging.info("In eveluate function")
    try:
        model_report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train,y_train)

            # prediction
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # R-squared score
            train_r2_score = r2_score(y_train, y_train_pred)
            test_r2_score = r2_score(y_test, y_test_pred)

            model_report[list(models.keys())[i]] = test_r2_score

        logging.info("Ready to return model report")
        return model_report
    except Exception as e:
        raise CustomException(e,sys)

    