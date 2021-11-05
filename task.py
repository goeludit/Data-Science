#Importing the packages required 
import pandas as pd 
import numpy as np 
from textblob import TextBlob 

#Reading the excel file
data = pd.read_csv("C:\\Users\\Udit\\Downloads\\task.csv", nrows = 50)


list_column = ['unique_id','raw_text','z']
new_data = pd.DataFrame(data, columns = list_column)

#Replace empty cells with NaN
x = new_data['raw_text']
x.replace('',np.nan, inplace = True)
print(x.shape)

#Dropping Empty Cells 
x.dropna(how = 'any', axis = 0, inplace = True)


for k in x:
    if k.isnumeric == True:
        x.drop(k)
    else:
        break

v = []

#Determinimg Sentiment value for the statement 
for i in x:
    g = TextBlob(i)

    v.append(g.sentiment.polarity)


    Sensitivty_value  = pd.Series(v, name='Sensitivty value ')
    Sensitivty_value  = pd.DataFrame(Sensitivty_value )
    #Joining the two dataframe 
    final = pd.concat((Sensitivty_value ,x), ignore_index = False, axis = 1)


print(final.head())



