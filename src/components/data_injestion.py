from email import header
from operator import index
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.components.data_transfomation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataInjestionconfig:
    train_data_path: str = os.path.join('artifact', 'train.csv')
    test_data_path: str = os.path.join('artifact', 'test.csv')
    raw_data_path: str = os.path.join('artifact', 'data.csv')


class DataInjestion:
    def __init__(self):
        self.injection_config = DataInjestionconfig()
    
    def initiate_data_injestion(self):
        logging.info("Enter in data injection methor or components")

        try:
            # load the raw data
            df = pd.read_csv(r'notebook\Dataset\StudentsPerformance.csv')
            logging.info("Reading Data as Datafream...")

            os.makedirs(os.path.dirname(self.injection_config.test_data_path),exist_ok=True)

            df.to_csv(self.injection_config.raw_data_path,index=False,header=True)
            logging.info("Data saved successfully at: {}".format(self.injection_config.raw_data_path))

            # split data into test set and train set
            logging.info("Splitting data into train and test sets...")
            train_set,test_set=train_test_split(df,test_size=0.30,random_state=42)

            train_set.to_csv(self.injection_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.injection_config.test_data_path,index=False,header=True)

            logging.info("Data saved successfully")

            return(
                self.injection_config.train_data_path,
                self.injection_config.test_data_path
            )
        except Exception as e:
            CustomException(e,sys)

if __name__ == "__main__":
    injection_config = DataInjestion()
    train_data_path,test_data_path=injection_config.initiate_data_injestion()
    data_transfomation = DataTransformation()
    train_data,test_data,_=data_transfomation.initiate_data_transformation(train_data_path,test_data_path)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_data,test_data))

        