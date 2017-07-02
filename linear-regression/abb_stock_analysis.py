import quandl, math
import numpy 
import pandas
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import json
import matplotlib.pyplot as plt
from matplotlib import style
from datetime import datetime 
import matplotlib.dates as mdates

stock_data = "ABB_stock.csv"
# columns: 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close*', 'Volume'

dataset = pandas.read_csv(stock_data)
dataset["Date"] = dataset["Date"].map(lambda x: datetime.strptime(str(x), '%Y-%m-%d'))
x = dataset["Date"]
y = dataset["Close"]

# plot
years = mdates.YearLocator()  
months = mdates.MonthLocator() 
yearsFmt = mdates.DateFormatter('%Y')

plt.gca().xaxis.set_major_locator(years)
plt.gca().xaxis.set_major_formatter(yearsFmt)
plt.gca().xaxis.set_minor_locator(months)  # months to be displayed on x axis  

plt.plot(x,y)

plt.gcf().autofmt_xdate()
plt.show()