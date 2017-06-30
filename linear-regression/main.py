import quandl, math
import numpy 
import pandas
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import json
import matplotlib.pyplot as plt
from matplotlib import style
import datetime

with open('secret.json') as secret_file:    
    secret_data = json.load(secret_file)

'''Using data from quandl repository, specifically stock value of GOOGLE'''
quandl.ApiConfig.api_key = secret_data["SECRET"]
df = quandl.get("WIKI/GOOGL")

# print(df.head())

'''Getting rid of redundant data. Taking into account only adjusted values.'''
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]


'''Looking for a set of data that may be useful for Machine Learning. Plain values do not
seem to be a perfect fit. Let's play with the values to obtain something meaningful:'''

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0

'''Daily change percentage:'''
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

'''Dataframe to continue playing with:'''
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
print(df.head())

'''In supervised learning, there are features and labels. 
The features are the descriptive attributes, and the label is value to predict or forecast.
Forecasting out the price, our label, the thing to predict, is actually the future price. 
As such, the features are actually: current price, high minus low percent, and the percent change volatility. 
The price that is the label shall be the price at some determined point the future.
'''

'''Defining the forecasting column - filling any NaN data with -99999 (common thing to do)'''
forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))

'''As it was decided, the features are a bunch of the current values, and the label shall be the price, 
in the future, where the future is 1% of the entire length of the dataset out. Let's assume all current columns are our 
features, so a new column with a simple pandas operation is added:'''

df['label'] = df[forecast_col].shift(-forecast_out)

'''It is a typical standard with machine learning in code to define X (capital x), as the features, 
and y (lowercase y) as the label that corresponds to the features. As such, features and label is defined:'''
X = numpy.array(df.drop(['label'], 1)) # X is basically all but 'label' ;)

'''
Pre-processing. Generally, it is desired to have features in machine learning to be in a range of -1 to 1. 
This may do nothing, but it usually speeds up processing and can also help with accuracy. 
Because this range is so popularly used, it is included in the preprocessing module of Scikit-Learn. 
To utilize this, it is possible to apply preprocessing.scale to X variable
'''

X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

'''Dropping all NaN from dataframe'''
df.dropna(inplace=True)

y = numpy.array(df['label'])

'''
Training and testing. The way this works is to take, for example, 75% of the data, 
and use it to train the machine learning classifier. 
Then, remaining 25% of the data is taken to test the classifier.
Thus, if the test is performed on the last 25% of the data, a sort of accuracy and reliability can be obtained, often called the confidence score. 
There are many ways to do this, but, probably the best way is using the build in cross_validation provided, 
since this also shuffles the data. The code to do this:
'''

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

'''
Defining the classifier. There are many classifiers in general available through Scikit-Learn, and even a few specifically for regression. 
For now, let's use Support Vector Regression from Scikit-Learn's svm package
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
'''

clf = svm.SVR()

'''Having classifier, it can be trained'''
clf.fit(X_train, y_train)

'''Having classifier trained, it can be tested'''
confidence = clf.score(X_test, y_test)
'''Variable confidence, would be an accurracy of the algorithm'''

print ("Accurracy of SVR algorithm: %s" % confidence)


'''For testing purpose lets check another classifier'''
clf = LinearRegression(n_jobs=-1)  # n_jobs determines threads to be used for running an algorithm 
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print ("Accurracy of LinearRegression algorithm: %s" % confidence)

'''
Let's have in mind that the data which is tried to be forecasted is not scaled like the training data was. 
The scale method scales based on all of the known data that is fed into it. Ideally, would be to scale both the training, testing, AND forecast/predicting data all together.
But it is not always possible or reasonable. 
Nevertheless should be done, if possible. As chosen data data is small enough and the processing time is low enough, 
all the data will be preprocesed and scaled at once

In many cases, it won't be able possible though. For gigabytes of data to train a classifier it may take days to train the classifier, 
so it wouldn't be desired thing to do this every single time when prediction is wanted. 
Thus, it may be considered NOT to scale anything, or scale the data separately. 
As usual, it's best to test both options and see which is best in a specific case.
'''

'''X_lately variable contains the most recent features, which is going to be predicted against. '''

forecast_set = clf.predict(X_lately)

'''Printing forecast data:'''
print("Forecasted data:")
print(forecast_set, confidence, forecast_out)


df['Forecast'] = numpy.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [numpy.nan for _ in range(len(df.columns)-1)]+[i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# https://pythonprogramming.net/pickling-scaling-machine-learning-tutorial/?completed=/forecasting-predicting-machine-learning-tutorial/