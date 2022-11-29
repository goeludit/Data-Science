import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale 
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics 
iris = pd.read_csv("C:\\Users\\Udit\\Desktop\\iris.csv", header=0)

cols2 = ['sepal.length', 'sepal.width', 'petal.length']

def scatter_plot(data, cols):
    sns.set_style('darkgrid')
    for col in cols:
        sns.set_style('darkgrid')
        sns.lmplot(data = iris,x = col ,y = 'petal.width' , hue ='variety', fit_reg = False)
        plt.title("Regression plot of " + 'Petal Width' + 'and' + col)
        plt.xlabel(col)
        plt.ylabel('Petal Width')
        

scatter_plot(iris,cols2 )

category = {'Setosa':0,'Versicolor':1,'Virginica':2}
np.set_printoptions(precision = 3, suppress=True)


iris['variety'] = [category[x] for x in iris['variety']]

num_cols = ['sepal.length','sepal.width','petal.length','petal.width']
num_cols2 = ['sepal.length','sepal.width','petal.length','petal.width','variety']

iris_scaled = scale(iris[num_cols])

iris_scaled = pd.DataFrame(iris_scaled)
iris_scaled['variety'] = iris['variety']



iris_split = train_test_split(np.asmatrix(iris_scaled), test_size = 55)

iris_test_label = np.ravel(iris_split[1][:,4])
iris_test_features = iris_split[1][:,:4]
iris_train_label = np.ravel(iris_split[0][:,4])
iris_train_features =iris_split[0][:,:4]

print(iris_train_features.shape)
print(iris_train_label.shape)
print(iris_test_features.shape)
print(iris_test_label.shape)


NB = GaussianNB()
NB.fit(iris_train_features,iris_train_label)


x = NB.predict(iris_test_features)
print(x)

print("Accuracy:",metrics.accuracy_score(iris_test_label,x))











