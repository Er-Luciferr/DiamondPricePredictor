import sys,os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')
            
            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)
            
            data_scaled= preprocessor.transform(features)
            
            pred=model.predict(data_scaled)
            return pred
        
        except Exception as e:
            logging.info('Exception occured in prediction')
            raise CustomException(e,sys)
        
        
class CustomData:
    def __init__(self,
                 carat:float,
                 depth:float,
                 clarity:str,
                 cut:str,
                 color:str,
                 table:float,
                 x:float,
                 y:float,
                 z:float):
        
        self.carat=carat
        self.depth=depth
        self.clarity=clarity
        self.cut=cut
        self.color=color
        self.table=table
        self.x=x
        self.y=y
        self.z=z
        
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict={
                'carat':[self.carat],
                'depth':[self.depth],
                'clarity':[self.clarity],
                'cut':[self.cut],
                'color':[self.color],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)