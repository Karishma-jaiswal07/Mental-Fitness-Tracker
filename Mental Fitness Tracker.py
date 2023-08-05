#!/usr/bin/env python
# coding: utf-8

# In[25]:


#Importing all the required libraries:
import numpy as np #for linear algebra 
import pandas as pd #for data processing and for the CSV files
import seaborn as sns 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[7]:


#Importing the data-set
data= pd.read_csv("mental-and-substance dataset.csv")
print("Importing Data is successfully done")


# In[8]:


#Checking whether the dataset is imported succesfully or not....
print("For this we will print first 10 values of the imported data-set")
data.head(10)


# In[9]:


print("We have correctly imported Data-set ")


# In[10]:


#Finding the missing values of the Dats-set:
data.isnull().sum()


# In[11]:


#Droping the columns 
data.drop('Code',axis=1 , inplace=True)


# In[12]:


data.head()


# In[13]:


data.size
data.shape


# In[16]:


#Exploratary Analysis
plt.figure(figsize=(12,6))
sns.heatmap(data.corr(),annot=True,cmap='Blues')


# In[27]:


mean=data['DALYs (Disability-Adjusted Life Years) - Mental disorders - Sex: Both - Age: All Ages (Percent)'].mean()
print(mean)


# In[28]:


df = data.copy()


# In[29]:


df.head()


# In[30]:


df.info()


# In[31]:


from sklearn.preprocessing import LabelEncoder


# In[32]:


l=LabelEncoder()
for i in df.columns:
    if df[i].dtype=="object":
        df[i]=l.fit_transform(df[i])


# In[33]:


X = df.drop('DALYs (Disability-Adjusted Life Years) - Mental disorders - Sex: Both - Age: All Ages (Percent)',axis=1)
y = df['DALYs (Disability-Adjusted Life Years) - Mental disorders - Sex: Both - Age: All Ages (Percent)']

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=2)


# In[37]:


#Applying the LINEAR REGRESSION:
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
lr = LinearRegression()
lr.fit(xtrain,ytrain)

# model evaluation for training set
ytrain_pred = lr.predict(xtrain)
mse = mean_squared_error(ytrain, ytrain_pred)
rmse = (np.sqrt(mean_squared_error(ytrain, ytrain_pred)))
r2 = r2_score(ytrain, ytrain_pred)

print("The model performance for training set")
print("--------------------------------------")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
ytest_pred = lr.predict(xtest)
mse = mean_squared_error(ytest, ytest_pred)
rmse = (np.sqrt(mean_squared_error(ytest, ytest_pred)))
r2 = r2_score(ytest, ytest_pred)

print("The model performance for testing set")
print("--------------------------------------")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))


# In[38]:


#Random FOREST REGRESSION:
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(xtrain, ytrain)

# model evaluation for training set
ytrain_pred = rf.predict(xtrain)
mse = mean_squared_error(ytrain, ytrain_pred)
rmse = (np.sqrt(mean_squared_error(ytrain, ytrain_pred)))
r2 = r2_score(ytrain, ytrain_pred)

print("The model performance for training set")
print("--------------------------------------")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
ytest_pred = rf.predict(xtest)
mse = mean_squared_error(ytest, ytest_pred)
rmse = (np.sqrt(mean_squared_error(ytest, ytest_pred)))
r2 = r2_score(ytest, ytest_pred)

print("The model performance for testing set")
print("--------------------------------------")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))


# In[ ]:




