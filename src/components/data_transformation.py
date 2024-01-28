from sklearn.impute import SimpleImputer ## Handling missing valoes
from sklearn.preprocessing import StandardScaler,OrdinalEncoder

##pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
import sys,os
from src.utils import save_object

from dataclasses import dataclass

@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
    
    
## Data Transformation config class    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()
        
    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
            
            ## Categorical and numerical columns
            categorical_cols = ['cut', 'color', 'clarity']
            numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']
            
            ##Define custom ranking for each ordinal variable
            cut_categories = [  'Fair', 'Good','Very Good','Premium','Ideal' ]
            color_categories=['D','E','F','G','H','I','J']
            clarity_categories=['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
            
            logging.info('Data Transformation pipeline Initiated')
            
            ### Numerical Pipeline 
            num_pipeline = Pipeline(
                steps= [
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
                
                ]
            )

            ### Categorical Pipeline 
            cat_pipeline = Pipeline(
                steps= [
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoding', OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                ('scaler', StandardScaler())
                ]
            )


            preprocessor = ColumnTransformer([
                ('numerical_pipe',num_pipeline,numerical_cols),
                ('categorical_pipe',cat_pipeline,categorical_cols)
            ])
            

            return preprocessor
            logging.info('Data Transformation Done')
            
            
            
        except Exception as e:
            logging.info('Exception occured in data transformation')
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        
        try:
            
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info('Reading of train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')
            
            logging.info('Obtaining preprocessing object')
            
            preprocessing_obj = self.get_data_transformation_object()
            
            target_column_name = 'price'
            drop_columns=[target_column_name,'id']
            
            ## features into independent and dependent features
            
            input_feature_train_df=train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            ##Data Transforation
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            logging.info('Applying preprocessor object on train and test set has been done')
            
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info('Processsor pickle in created and saved')
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            
            raise CustomException(e,sys)
 