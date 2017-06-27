import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
'''
Pandas provides facility to read csv data easily. Additionally first row names were
introduced, for sake of data readibility
'''

url = "iris.data.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

'''
Let's see dimensions of the dataset (150,5)
'''
print(dataset.shape)

'''
Having a look at first 20 rows
'''
print(dataset.head(20))

'''
Descriptions shows basic statistical summary for datasets.
We get: count, mean, std, min, 25%, 50%, 75%, max.
'''
print(dataset.describe())

'''
This will show how many items belongs to each group (as
we are working on three spieces of irises we get 3 different groups,
each containing 50 elements.
'''
print(dataset.groupby('class').size())

'''
Presents distribution of the data from dataset.
'''
dataset.plot(kind='box',subplots=True,layout=(2,2),sharex=False, sharey=False)
plt.show()

'''
Another way to present distribution. Histograms allows to
visually catch Gaussian distribution.
'''
dataset.hist()
plt.show()

'''
Scatter plots helps to detect correlation between pairs of attributes.
'''
scatter_matrix(dataset)
plt.show()

'''
Learning the parameters of a prediction function and testing it on the same data
is a methodological mistake: a model that would just repeat the labels of the samples 
that it has just seen would have a perfect score but would fail to predict anything 
useful on yet-unseen data. This situation is called overfitting. 
To avoid it, it is common practice when performing a (supervised) machine learning 
experiment to hold out part of the available data as a test set X_test, y_test. 

Here the loaded dataset is split into two, 80% of which is used to train the models and 
20% that will be held back as a validation dataset.
'''
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

'''
The plan is to split dataset into 10 parts, train (run algorithms) on 9 and test on 1. 

The metric of ‘accuracy‘ is used to evaluate models. This is a ratio of the number of correctly predicted instances
in divided by the total number of instances in the dataset multiplied by 100 to give a percentage 
(e.g. 95% accurate). Scoring variable will be used when build is run and to evaluate each model next.
'''

scoring = 'accuracy'

'''
Let’s evaluate 6 different algorithms:

Logistic Regression (LR)
Linear Discriminant Analysis (LDA)
K-Nearest Neighbors (KNN).
Classification and Regression Trees (CART).
Gaussian Naive Bayes (NB).
Support Vector Machines (SVM).

This is a good mixture of simple linear (LR and LDA), nonlinear (KNN, CART, NB and SVM) algorithms. 
We reset the random number seed before each run to ensure that the evaluation of each algorithm is performed using exactly the same data splits. 
It ensures the results are directly comparable.
'''
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []

'''
LR: 0.966667 (0.040825)
LDA: 0.975000 (0.038188)
KNN: 0.983333 (0.033333)
CART: 0.975000 (0.038188)
NB: 0.975000 (0.053359)
SVM: 0.981667 (0.025000)

From the output it can be set that KNN is the most accurate.
'''

for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


'''Printing the plot of the model evaluation. It results and compare the spread and the mean accuracy of each model. 
There is a population of accuracy measures for each algorithm because 
each algorithm was evaluated 10 times (10 fold cross validation).'''

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

'''As KNN was the most accurate, the evaluation must be performed.
It is valuable to keep a validation set just in case you made a slip during training, such
as overfitting to the training set or a data leak. 
Both will result in an overly optimistic result'''

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)

'''3 out of 27 were mistaken. Therefore accuracy = 0.9'''
print(accuracy_score(Y_validation, predictions))

'''https://en.wikipedia.org/wiki/Confusion_matrix
Diagonal represents valid guesses. All other values are mistakes.'''
print(confusion_matrix(Y_validation, predictions))

'''Summary below'''
print(classification_report(Y_validation, predictions))