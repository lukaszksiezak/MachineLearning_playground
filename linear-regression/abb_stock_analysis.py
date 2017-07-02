import quandl, math
import numpy 
import pandas
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
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
import json
import matplotlib.pyplot as plt
from matplotlib import style
from datetime import datetime 
import matplotlib.dates as mdates

stock_data = "ABB_stock.csv"
# columns: 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close*', 'Volume'

dataset = pandas.read_csv(stock_data)

X = dataset["Close"]
Y = a = [0 for x in range(len(X))]

validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

scoring = 'accuracy'

models = []
models.append(('LR', LinearRegression()))
results = []
names = []

for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# plot
# dataset["Date"] = dataset["Date"].map(lambda x: datetime.strptime(str(x), '%Y-%m-%d'))
# x = dataset["Date"]
# y = dataset["Close"]

# years = mdates.YearLocator()  
# months = mdates.MonthLocator() 
# yearsFmt = mdates.DateFormatter('%Y')

# plt.gca().xaxis.set_major_locator(years)
# plt.gca().xaxis.set_major_formatter(yearsFmt)
# plt.gca().xaxis.set_minor_locator(months)  # months to be displayed on x axis  

# plt.plot(x,y)

# plt.gcf().autofmt_xdate()
# plt.show()