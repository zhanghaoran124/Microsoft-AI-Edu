import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')
model = SVC(kernel='linear')

# 导入数据集
data_set = read_csv('iris.csv')

# 分离数据集
array = data_set.values  

# 将dataset里的数值转化成array
X = array[:, 0:4]
Y = array[:, 4]
validation_size = 0.2   

# 随机种子
seed = 10
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

# 算法审查
models = {}
models['LR'] = LogisticRegression()
models['LDA'] = LinearDiscriminantAnalysis()
models['KNN'] = KNeighborsClassifier()
models['CART'] = DecisionTreeClassifier()
models['NB'] = GaussianNB()
models['SVM'] = SVC()

# 评估算法
print('各种算法的精确度比较：')
results = []
for key in models:   
    kfold = KFold(n_splits=10, random_state=seed, shuffle=True)   
    cv_results = cross_val_score(models[key], X_train, Y_train, cv=kfold, scoring='accuracy')   
    results.append(cv_results)
    print('%s: %f (%f)'%(key,cv_results.mean(),cv_results.std()))

# 使用评估数据集评估算法
print('')
print('利用SVM算法的训练结果：')
svm = SVC()
svm.fit(X=X_train,y=Y_train)
predictions = svm.predict(X_validation)
print('精确度为：',accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))