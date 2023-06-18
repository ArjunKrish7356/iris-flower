import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


df=pd.read_csv('Datasets\IRIS flower dataset.csv')
# Visualize the whole dataset
data = df.values
# Separate features and target  
X=data[:,0:4]
Y=data[:,4]
columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels']

sns.pairplot(df,hue=columns)
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Support vector machine algorithm
from sklearn.svm import SVC
svn = SVC()
svn.fit(X_train, y_train)

# Predict from the test dataset
predictions = svn.predict(X_test)

og=pd.DataFrame(X_test)
og['new'] = predictions

column_names = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels', 'new']
og = og.reindex(columns=column_names)

# Assigning actual data to the other columns
og['Sepal length'] = X_test[:, 0]
og['Sepal width'] = X_test[:, 1]
og['Petal length'] = X_test[:, 2]
og['Petal width'] = X_test[:, 3]
og['Class_labels'] = y_test[:]

print(og)





# Calculate the accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))



sns.pairplot(df,hue=columns)
plt.show()