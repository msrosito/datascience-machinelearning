import pandas as pd
import numpy as np

input_dir = '../data/'
output_dir = './resultsPD/'

# Definition of dataframes #####################################################

# Dictionary
df = pd.DataFrame({'a': [10, 20, 30], 'b': [4, 5, 6], 'c': [7, 8, 9]}, 
                   index = [1, 2, 3])
print(df)

mydict = {'a': [10, 20, 30], 'b': [4, 5, 6], 'c': [7, 8, 9]}
df = pd.DataFrame.from_dict(mydict)
print(df)

# Matrix - Numpy
df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], index = ['ind1', 'ind2'], 
                   columns = ['a', 'b', 'c'])
print(df)

# From csv
df = pd.read_csv(input_dir + 'data_example_LR.csv')
print(df)
# two columns = x, y

# Access to data ###############################################################

# By index
print(df.iloc[10: 15]) # rows 10 to 14
print(df.iloc[:, [1]]) # column 1
print(df.iloc[[1, 3], [0]]) # rows 1 and 3, column 0

# By label
print(df.loc[0: 2,  'x']) # first three rows, column 0
column0 = df['x']
print((column0)) # column 0

# Single values
print(df.iat[1, 1]) # row 1, column 1
print(df.at[4, 'y']) # row 4, column 1

# Add data #####################################################################

# Column
df['newcol'] = np.zeros(len(df))
df['x + y'] = df.x + df.y
print(df)

# Row
df.loc[len(df)] = [1, 2, 3, 4]
print(df)
newrow = {'x': 500, 'y': 100, 'newcol': 300, 'x + y': 600}
df = df.append(newrow, ignore_index = True)
print(df)

# Summarization ################################################################
print('Information about the data frame')
print('Number of rows: ', len(df))
print('Shape: ', df.shape)
print('Number of distinct values of newcol: ', df['newcol'].nunique())
print('Number of repetitions of each value in newcol: ', df['newcol'].value_counts())
print('Sumamary', df.describe())

# Save dataframe
df.to_csv(output_dir + 'df.csv')
df.to_csv(output_dir + 'df-noindex.csv', index = False) # remove indexes
