# Given a list of paths of CSV files (six columns: date, open, max, min, close, vol)
# return a list where each element is associated to a different file and contains two dataframes:
# First dataframe. Column 1: date of the day with the highest vol. Column 2: highest vol.
# Second dataframe: Column 1: date of the days with the highest close. Column 2:
# highest close price. If the same closing price ocurred on more than one day,
# the function should include them all 

import numpy as np
import pandas as pd
import os

## Function definition #########################################################

def solution(files):
    res = []
    for f in files:
        df = pd.read_csv(f)
        df['year'] = pd.DatetimeIndex(df['date']).year
        df_aux = df.groupby(by = 'year').rank(method = 'min', ascending = False)
        df['rank_vol'] = df_aux['vol']
        df['rank_close'] = df_aux['close']
        
        df1 = pd.DataFrame({}) # first dataframe
        df1['date'] = df.query('rank_vol == 1')['date']
        df1['vol'] = df.query('rank_vol == 1')['vol']
        df1 = df1.reset_index(drop = True)
        
        df2 = pd.DataFrame({}) # second dataframe
        df2['date'] = df.query('rank_close == 1')['date']
        df2['close'] = df.query('rank_close == 1')['close']
        df2 = df2.reset_index(drop = True)
        
        res.append([df1, df2])
        
    return res

## Main ########################################################################
    
input_dir = './data/'

files = os.listdir(input_dir)
files = [str(input_dir) + f for f in files] 

print(files)
print(solution(files))    
