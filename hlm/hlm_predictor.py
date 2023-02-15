import numpy as np
import pandas as pd
import time
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from pandas import DataFrame
from numpy import array

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, ".."))

from hlm import hlm_gcnn_scaler, hlm_gcnn_model, hlm_gcnn_model_version
# from hlm import hlm_gcnn_scaler, hlm_gcnn_model
from base.gcnn import GcnnBase


# class HLMPredictor(GcnnBase):
#     """
#     Makes HLM stability preditions

#     Attributes:
#         df (DataFrame): DataFrame containing column with smiles
#         smiles_column_index (int): index of column containing smiles
#         predictions_df (DataFrame): DataFrame hosting all predictions
#     """

#     def __init__(
#         self, 
#         kekule_smiles: array = None, 
#         smiles: array = None
#         ):
#         """
#         Constructor for HLMPredictor class

#         Parameters:
#             kekule_smiles (Array): numpy array of RDkit molecules
#         """
#         GcnnBase.__init__(
#             self, 
#             kekule_smiles, 
#             column_dict_key='Predicted Class (Probability)', 
#             columns_dict_order = 1, 
#             smiles=smiles
#             )

#         self._columns_dict['Prediction'] = {
#             'order': 2,
#             'description': 'class label',
#             'isSmilesColumn': False
#         }

#         self.model_name = 'hlm'

#     def get_predictions(self) -> DataFrame:
#         """
#         Function that calculates consensus predictions

#         Returns:
#             Predictions (DataFrame): DataFrame with all predictions
#         """

#         if len(self.kekule_smiles) > 0:

#             start = time.time()
#             gcnn_predictions, gcnn_labels = self.gcnn_predict(hlm_gcnn_model, hlm_gcnn_scaler)
#             end = time.time()
#             print(f'HLM: {end - start} seconds to predict {len(self.predictions_df.index)} molecules')

#             self.predictions_df['Prediction'] = pd.Series(
#                 pd.Series(np.where(gcnn_predictions>=0.5, 'unstable', 'stable'))
#             )
            
#         return self.predictions_df


# from rlm 
class HLMPredictor(GcnnBase):
    """
    Makes HLM stability preditions

    Attributes:
        df (DataFrame): DataFrame containing column with smiles
        smiles_column_index (int): index of column containing smiles
        predictions_df (DataFrame): DataFrame hosting all predictions
    """

    def __init__(self, kekule_smiles: array = None, smiles: array = None):
        """
        Constructor for HLMPredictor class

        Parameters:
            kekule_smiles (Array): numpy array of RDKit molecules
        """

        GcnnBase.__init__(
            self, 
            kekule_smiles, 
            column_dict_key='Predicted Class (Probability)', 
            columns_dict_order = 1, 
            smiles=smiles
            )

        self._columns_dict['Prediction'] = {
            'order': 2,
            'description': 'class label',
            'isSmilesColumn': False
        }

        self.model_name = 'hlm'
        self.model_version = hlm_gcnn_model_version

    def get_predictions(self) -> DataFrame:
        """
        Function that calculates consensus predictions

        Returns:
            Predictions (DataFrame): DataFrame with all predictions
        """

        if len(self.kekule_smiles) > 0:

            start = time.time()
            gcnn_predictions, gcnn_labels = self.gcnn_predict(hlm_gcnn_model, hlm_gcnn_scaler)
            end = time.time()
            print(f'HLM: {end - start} seconds to predict {len(self.predictions_df.index)} molecules')

            self.predictions_df['Prediction'] = pd.Series(
                pd.Series(np.where(
                    gcnn_predictions>=0.5, 
                    'unstable', 
                    'stable'
                    )
                )
            )
        return self.predictions_df

    def get_model_version(self) -> str:
        return self.model_version