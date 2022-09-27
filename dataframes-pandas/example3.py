import pandas as pd

# Definition of dataframes #####################################################

df = pd.DataFrame({'a': [10, 20, 30, 0, 10, 10], 'b': [4, 5, 6, 7, 3, 8], 
'd': [40, 50, 10, 3, 3, 3]})
df['ab'] = df['a'] * df['b']

print('Dataframe 1: \n', df)

# Filtrering ####################################################################
print('Rows having a > 15: \n', df.query('a > 15'))
print('Rows having a > 15 and ab = 0: \n', df.query('a > 15 and ab == 0')) # empty

# Group data ####################################################################
gb = df.groupby(by = 'a') # group by object

newdf1 = gb.max()
print('Maximum values of each column for each value of a: \n', newdf1)
newdf2 = gb.agg({'b': ['max', 'sum'], 'ab' : ['max']})
print('Maximum and sum of column b and maximum of column ab for each value of a: \n', newdf2)

gb = df.groupby(by = ['a', 'd'])

newdf3 = gb['b'].count().reset_index(name = 'count')
print('Number of elements in each group: \n', newdf3)

# Window functions ##############################################################
newdf4 = df.rolling(3).sum()
print('Rolling sum for each column, window size = 2: \n', newdf4)

col1 = df['a'].rolling(3).max()
col2 = df['b'].rolling(3).max()
newdf5 = pd.DataFrame({'Max a': col1, 'Max b': col2})
print('Rolling maximum for columns a and b, window size = 2: \n', newdf5)  
