import os
import sys

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