import os
import sys
from src.exception import CustomException
from src.logger import logging
import dill

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