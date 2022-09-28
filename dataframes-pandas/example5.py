# Given a string consisting on substrings of the format <<photoname>>.<<extension>>, 
# <<city_name>>, <<yyyy-mm-dd hh:mm:ss>> separated by newline characters return 
# a string consisting on substrings with the format <<city_name>><<number>>.<<extension>>
# separated by newline characters. The number should sort the substrings according 
# to the date grouped by city and should have the same lenght for each city
# The photos must be in the same order as in the input and the total number of 
# substrings is 99.

import pandas as pd

## Function definition #########################################################

def solutions(S):
    S = S.split('\n')
    M = len(S)
    df = pd.DataFrame(columns = ['name', 'city', 'date'], 
    data = [row.split(', ') for row in S])
    df['extension'] = [df.loc[i, 'name'].split('.')[1] for i in range(M)] # first dataframe
    gb = df.groupby(by = 'city')
    tot = gb['name'].count().reset_index(name = 'n_city') # second dataframe
    n_order = gb.rank() # third dataframe

    res = ''

    for i in range(M):
        city_name = str(df.at[i, 'city'])
        extension = str(df.at[i, 'extension'])
        num = int(n_order.at[i, 'date'])
        n_city = tot.query('city == @city_name').iat[0, 1]    
        if n_city >= 10 and num <= 9:
            num = '0' + str(num)
        name = city_name + str(num) + '.' + extension
        res = res + name + '\n'
    return res

## Main ########################################################################
    
S = 'photo.jpg, Warsaw, 2013-09-05 14:08:15\njohn.png, London, 2015-06-20 15:13:22\nmyFriends.png, Warsaw, 2013-09-05 14:07:13\nEiffel.jpg, Paris, 2015-07-23 08:03:02\npisatowor.jpg, Paris, 2015-07-22 23:59:59\nBOB.jpg, London, 2015-08-05 00:02:03\nnotredame.png, Paris, 2015-09-01 12:00:00\nme.jpg, Warsaw, 2013-09-06 15:40:22\na.png, Warsaw, 2016-02-13 13:33:50\nb.jpg, Warsaw, 2016-01-02 15:12:22\nc.jpg, Warsaw, 2016-01-02 14:34:30\nd.jpg, Warsaw, 2016-01-02 15:15:01\ne.png, Warsaw, 2016-01-02 09:49:09\nf.png, Warsaw, 2016-01-02 10:55:32\ng.jpg, Warsaw, 2016-02-29 22:13:11'

print(solutions(S))         
