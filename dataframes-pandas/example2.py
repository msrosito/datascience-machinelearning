import pandas as pd

# Definition of dataframes #####################################################

df1 = pd.DataFrame({'a': [10, 20, 30], 'b': [4, 5, 6], 'd': [40, 50, 10]})
df2 = pd.DataFrame({'a': [10, 20, 35, 6], 'c': [7, 8, 9, 20]})

print('Dataframe 1: \n', df1)
print('Dataframe 2: \n', df2)

# Merge ########################################################################

print('Left: \n', pd.merge(df1, df2, how = 'left', on = 'a')) # equivalent to LEFT JOIN MySQL
print('Right: \n', pd.merge(df1, df2, how = 'right', on = 'a')) # equivalent to RIGHT JOIN MySQL
print('Inner: \n', pd.merge(df1, df2, how = 'inner', on = 'a')) # equivalent to JOIN MySQL
print('Outer: \n', pd.merge(df1, df2, how = 'outer', on = 'a'))

# Reshape ######################################################################

print('Original shape dataframe 1: ', df1.shape)
print('Original shape dataframe 2: ', df2.shape)

# Row into columns
resh = pd.melt(df1)
print('Row into columns dataframe 1: \n', resh)
print('New shape dataframe 1: ', resh.shape)

# Concatenate
conc1 = pd.concat([df1, df2])
conc2 = pd.concat([df1, df2], axis = 1)
print('Append rows: \n', conc1)
print('Shape concatenation: ', conc1.shape)
print('Append columns: ', conc2)
print('Shape concatenation: \n', conc2.shape)
