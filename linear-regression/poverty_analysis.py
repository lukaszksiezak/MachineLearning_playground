# Attempt to perform supervised learning using linear regression

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
df = quandl.get("WPOV/POL_SI_POV_NOP1") # data representing number of poor ar 5PLN per day in Poland 

print df.head()
