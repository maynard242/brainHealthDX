# Program to load model and predict class
# Input: xgb_model.py %1 where %1 is a csv file of features
# Output: array of predictions

import os
import sys
import xgboost as xgb
import pandas as pd
import json
import pickle as pkl

# Load model
xgb_model = pkl.load(open('xgb_model.pkl','rb'))

# Read csv file
inputcvs = pd.read_json(str(sys.argv[1]))
                       
# Process data
load_data_matrix = xgb.DMatrix(inputcvs)
print(xgb_model.predict(load_data_matrix))                       
                       
                                     