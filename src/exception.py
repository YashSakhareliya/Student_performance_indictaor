import sys
from src.logger import logging

def error_message_details(error, error_details:sys):
    '''
    This function takes an error message and the system error details (sys) as input.
    returns an error message
    '''
    _,_,exc_tb = error_details.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
    file_name,exc_tb.tb_lineno,str(error)
    )
    return error_message


class CustomException(Exception):
    def __init__(self,error_message,error_details:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error=error_message,error_details=error_details)
    
    def __str__(self):
        return self.error_message
    

if __name__ == "__main__":
    try:
        a=1/0
    except Exception as e:
        logging.info("Division by zero")
        raise CustomException(e,sys)