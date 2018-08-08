import os
import sys
import xgboost as xgb
import pandas as pd
import numpy as np
import json
import pickle as pkl

# Load model
xgb_model = pkl.load(open('xgb_model.pkl','rb'))

# Load distribution
load_dfc = pd.read_csv('cognitive_health.csv')
load_dfp = pd.read_csv('physical_health.csv')
load_dfl = pd.read_csv('living_health.csv')

# cognitive health
def compute_cog(d):
    """
    Args:
    d: dict

    Returns:
    percentage

    """
    accuracy=(d['22']+d['25']+d['23']+d['24'])/27
    return round(accuracy*100, 3)

# physical health
def compute_physical(d):
    """
    Args:
    d: dic

    Returns:
    percentage

    """
    # binary questions
    binary_ques=[5, 7, 8, 10, 12, 20]
    binary_score=[d[str(b)] for b in binary_ques].count(1)
    if d['17']==5:
        binary_score+=1

    # numeric questions
    numeric_ques=[4, 6, 9, 11, 13, 14, 15, 16, 18, 21]
    numeric_score=sum([d[str(n)] for n in numeric_ques])

    # final score
    final_score=(binary_score+numeric_score)/72
    return (round((1-final_score)*100, 3))

# living health
def compute_living(d):
    """
    Args:
    d: dict

    Returns:
    percentage

    """
    score=0
    score+=[d[str(k)] for k in range(26,41)].count(1)
    if d['41']>=1:
        score+=1
    if d['42']!=1:
        score+=1
    final_score=score/17
    return round((1-final_score)*100, 3)


def api_predict(data):

    # Transform data
    data['22'] = data['22'][0] 
    data['25'] = data['25'][0]
    temp = 0
    for i in range(5):
        if data['24'][i] == (100-47-7-i*7):
            temp += 1
    data['24'] = temp
    data['total_recall'] = data['22'] + data['25']

    # Calculate scores
    cog_score = compute_cog(data)
    phy_score = compute_physical(data)
    liv_score = compute_living(data)

    # Create panda
    load_df = pd.DataFrame(data, index=[0])
    #load_df = pd.read_json(inputfile)


    # Make sure order is consistent with training
    json_2_df = load_df.loc[:,['23', '25', '1', '2', '3', '4',
       '5', '6', '7', '8', '9', '10', '11', '12',
       '13', '14', '15', '16', '17', '18', '19', '20',
       '21', '26', '27', '28', '29', '30', '31',
       '32', '33', '34', '35', '36', '37', '38', '39',
       '40', '41', '42', '22', '24', 'total_recall'
        ]]

    # Load data
    load_data_matrix = xgb.DMatrix(json_2_df)

    # Run prediction
    res = xgb_model.predict(load_data_matrix)

    # Calculate LW2009 score
    lw2009 = json_2_df['total_recall'][0] + json_2_df['23'][0] + json_2_df['24'][0]
    if lw2009 > 12:
        lw = 0
    else:
        lw = 1


    # Add in scores

    # Access info for scores
    i = 0
    if json_2_df['2'][0] < 50:
        i = 1
    elif json_2_df['2'][0] >= 50 and json_2_df['2'][0] < 60:
        i = 2
    elif json_2_df['2'][0] >= 60 and json_2_df['2'][0] < 70:
        i = 3
    elif json_2_df['2'][0] >= 70 and json_2_df['2'][0] < 80:
        i = 4
    elif json_2_df['2'][0] >= 80 and json_2_df['2'][0] < 90:
        i = 5
    elif json_2_df['2'][0] >= 90 and json_2_df['2'][0] < 100:
        i = 6
    else:
        i = 7

    # Given an age group

    cogs = load_dfc.loc[load_dfc['group']==i, ['mean', '25%', '50%', '75%',  'max',]].values.tolist()[0]
    cogs.append(cog_score)

    phys = load_dfp.loc[load_dfp['group']==i, ['mean', '25%', '50%', '75%',  'max',]].values.tolist()[0]
    phys.append(phy_score)

    livs = load_dfl.loc[load_dfl['group']==i, ['mean', '25%', '50%', '75%',  'max',]].values.tolist()[0]
    livs.append(liv_score)

    # Calculate Predicted Class, using a 0.5 hurdle - may need to change

    if res[0] > 0.5:
        pred_class = 1
    else:
        pred_class = 0

    # Calculate the Aggregate Dementia Risk Score:
    # Score = 1 if mixed results
    agg_score = 1
    # Score = 2 if both dementia class
    if pred_class == 1 & lw == 1:
        agg_score = 2
    # Score = 0 if both normal class
    if pred_class ==0 & lw == 0:
        agg_score = 0

    #print(agg_score,cog_score, cogs[2], phy_score, phys[2], liv_score, livs[2])

    pref_str = ','.join(map(str,[agg_score, cog_score, cogs[0], phy_score, phys[0], liv_score, livs[0]]))

    original_str = '{ "Predicted": ' + str(res[0]) + ', "LW2009": ' + str(lw2009) + ', "LWClass": ' + str(lw) + ', "Cog": ' + str(cogs) + ', "Phys": ' + str(phys) + ', "Livs": ' + str(livs) + '}'

    return pref_str
