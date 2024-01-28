import os
import sys
#sys.path.insert(0, '../src')
import pandas as pd
from src.logger import logging
from src.exception import CustomException

#import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

## Intitialize the data ingestion configuration

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts' , 'train.csv')
    test_data_path:str=os.path.join('artifacts' , 'test.csv')
    raw_data_path:str=os.path.join('artifacts' , 'raw.csv')

## create data ingestion class

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion method starts')

        try:
            df=pd.read_csv(os.path.join('notebook/data','gemstone.csv'))
            logging.info('dataset read as pandas dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
    
            logging.info("Raw data is created")

            train_set,test_set=train_test_split(df,test_size=.30,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Train and Test datasets are created /n Ingestion of data DONE")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )


        except Exception as e:
            logging.info('Exception occured at Data ingestion stage')
            raise CustomException(e,sys)
        