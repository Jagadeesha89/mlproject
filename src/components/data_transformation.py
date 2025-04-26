import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import dill

from src.exception import CustomException
from src.logger import logging
from src.utlis import save_object

@dataclass
class DataTranformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_trnformation_config=DataTranformationConfig()

    def get_transformer_object(self):
        try:
            numerical_colmumns=['writing_score','reading_score']
            categorical_features=[
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            num_pipline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler(with_mean=False))
                ]
            )

            cat_pipline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy='most_frequent')),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical columns standard scaling done")
            logging.info("Categorical columns encoding completed")

            preprocessor=ColumnTransformer(
                [
                ("num_pipline",num_pipline,numerical_colmumns),
                ("cat_pipline",cat_pipline,categorical_features)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_tranformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing obeject")

            preprocessing_obj=self.get_transformer_object()

            target_columns_name="math_score"
            numerical_columns = ['writing_score','reading_score']

            input_feature_train_df=train_df.drop(columns=[target_columns_name],axis=1)
            target_feature_train_df=train_df[target_columns_name]

            input_feature_test_df=test_df.drop(columns=[target_columns_name],axis=1)
            target_feature_test_df=test_df[target_columns_name]
            
            logging.info(f"applying preprocessing object on training dataframe")

            input_features_train_array=preprocessing_obj.fit_transform(input_feature_train_df)
            input_features_test_array=preprocessing_obj.fit_transform(input_feature_test_df)

            train_arr=np.c_[
                input_features_train_array,np.array(target_feature_train_df)
            ]

            test_arr=np.c_[
                input_features_test_array,np.array(target_feature_test_df)
            ]

            logging.info("saved preprocessing object")

            save_object(
                file_path=self.data_trnformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_trnformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            raise CustomException(e,sys)
            

        
