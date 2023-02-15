import os
import sys
from tqdm import tqdm
from datetime import datetime

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, ".."))

# from chemprop.chemprop.utils import load_checkpoint, load_scalers
from predictors.utilities.utilities import load_gcnn_model
from os import path

hlm_model_file_url = os.path.abspath(os.path.join(root, '../../../checkpoints/gcnn_model.pt'))
hlm_model_file_path = os.path.abspath(os.path.join(root, '../../../checkpoints/gcnn_model.pt'))



print(f'Loading HLM graph convolutional neural network model', file=sys.stdout)
os.makedirs('../../checkpoints', exist_ok=True)
hlm_gcnn_scaler, hlm_gcnn_model = load_gcnn_model(hlm_model_file_path, hlm_model_file_url)

del hlm_model_file_url
del hlm_model_file_path

print(f'Finished loading HLM models', file=sys.stdout)







# # from rlm 
# hlm_model_file_url = os.path.abspath(os.path.join(root, '../../../checkpoints'))
# hlm_model_file_path = os.path.abspath(os.path.join(root, '../../../checkpoints'))


# def load_gcnn_model():
#     os.makedirs('../../checkpoints', exist_ok=True)
#     print(f'Loading HLM graph convolutional neural network model', file=sys.stdout)
#     hlm_gcnn_scaler_path = f'{hlm_model_file_path}/gcnn_model.pt'
#     hlm_gcnn_scaler, _ = load_scalers(hlm_gcnn_scaler_path)
#     hlm_gcnn_model = load_checkpoint(hlm_gcnn_scaler_path)

#     model_timestamp = datetime.fromtimestamp(os.path.getctime(hlm_gcnn_scaler_path)).strftime('%Y-%m-%d') # get model file creation timestamp
#     hlm_gcnn_model_version = 'hlm_' + model_timestamp # generate a model timestamp

#     return hlm_gcnn_scaler, hlm_gcnn_model, hlm_gcnn_model_version
# hlm_gcnn_scaler, hlm_gcnn_model, hlm_gcnn_model_version = load_gcnn_model()

# print(f'Finished loading HLM model files', file=sys.stdout)


