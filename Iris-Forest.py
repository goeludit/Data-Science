from matplotlib import pyplot as plt 
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import scale
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

iris = pd.read_csv("C:\\Users\\Udit\\Desktop\\iris.csv", header = 0)
print(iris.columns)
array = np.array(iris['variety'])
np.set_printoptions(2,None)


color = {'Setosa':'red','Versicolor':'green','Virginica':'blue'}
l = {'Setosa':'Setosa','Versicolor':'Versicolor','Virginica':'Virginica'}
def plt_scatter (iris,cols,col_x = 'petal.width'):
 for col in cols:
    
    plt.scatter(data = iris, x = col_x, y = col, c = iris['variety'].apply(lambda x : color[x]))
    plt.xlabel(col_x)
    plt.ylabel(col)
    plt.title("Regression Chart ")




iris_features = ['sepal.length','sepal.width','petal.length']
plt_scatter(iris,iris_features)



num_cols = ['sepal.length','sepal.width','petal.length','petal.width']
iris_scaled = scale(iris[num_cols])
iris_scaled = pd.DataFrame(iris_scaled, columns = num_cols)


iris_scaled['variety'] = iris['variety']


levels = {'Setosa':0, 'Versicolor':1, 'Virginica':2}
iris_scaled['variety'] = [levels[x] for x in iris['variety']]

from sklearn.model_selection import train_test_split


np.random.seed(3456)
iris_split = train_test_split(np.asmatrix(iris_scaled), test_size = 75)
iris_train_features = iris_split[0][:, :4]
iris_train_labels = np.ravel(iris_split[0][:, 4])
iris_test_features = iris_split[1][:, :4]
iris_test_labels = np.ravel(iris_split[1][:, 4])
print(iris_train_features.shape)
print(iris_train_labels.shape)
print(iris_test_features.shape)
print(iris_test_labels.shape)


Forest_classifier = RandomForestClassifier(10,criterion = 'entropy')
Forest_classifier.fit(iris_train_features,iris_train_labels)


iris_final = pd.DataFrame(iris_test_features)
iris_final['Prediciton'] = Forest_classifier.predict(iris_test_features)

print(sum(iris_final['Prediciton']))
print(iris_test_labels)

iris_final['correct'] = [1 if x == z else 0 for x,z in zip(iris_final['Prediciton'],iris_test_labels)]
accuracy = sum(iris_final['correct'])/iris_test_labels.shape[0] * 100
print(accuracy)