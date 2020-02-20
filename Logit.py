#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome'] # variables for one-hot encoding


# In[2]:


logitDf = pd.read_csv('Logistic_regression.csv') # load data


# In[3]:




# In[4]:


logitDfCat = logitDf[cat_vars] # dataframe of categorical data


# In[5]:


logitDfCatDummy = pd.get_dummies(logitDfCat) # cat data to dummy variables


# In[6]:



# In[7]:


logitDfNum = logitDf.drop(cat_vars, axis=1) # # dataframe of numerical data


# In[8]:



# In[9]:


LogitDfNew = pd.concat([logitDfNum, logitDfCatDummy], axis=1).reindex(logitDfNum.index) # new dataframe


# In[10]:


x_all = LogitDfNew.drop('y', axis=1).fillna(LogitDfNew.median()) # X for logit model


# In[11]:


x_all


# In[12]:


y = LogitDfNew[['y']] # y label


# In[13]:


y


# In[14]:


def sigmoid(x_arr):
    return 1.0/(1+np.exp(-x_arr))


# In[27]:


def logistic_regression(x_arr, y_arr, alpha=.001, max_iter=1000): # return weights(coefficients)
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr)
    weight = np.ones((x_mat.shape[1],1))
    for k in range(max_iter):
        h = sigmoid(x_mat * weight)
        error = (y_mat - h)
        weight = weight + alpha * x_mat.T * error
    return weight


# In[16]:


weight = logistic_regression(x_all,y) # test the function


# In[19]:


import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats


# In[20]:


from imblearn.over_sampling import SMOTE # SMOTE to deal with imbalance data
sm = SMOTE(random_state=42)
X, y = sm.fit_resample(x_all, y)

# havent split x y into test and train yet. But sklearn has the api. 

# rest still undone. 