#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# # Loading Dataset

# In[3]:


df=pd.read_csv("50_Startups.csv")
df


# In[4]:


df.head()


# In[5]:


df.dtypes


# # Checking for Null Values

# In[6]:


df.isnull().sum()


# # Data Visualization

# # Boxplot:

# In[7]:


sns.boxplot(x=df['Profit'])


# In[8]:


sns.boxplot(x=df['Marketing Spend'])


# In[9]:


sns.boxplot(x=df['Administration'])


# In[10]:


sns.boxplot(x=df['R&D Spend'])


# # Correlated Heatmap:

# In[11]:


hm=sns.heatmap(data=df.corr())


# In[12]:


corr=df[df.columns[0:]].corr()['Profit'][:]
plt.plot(corr)
plt.xticks(rotation=90)
plt.show()


# # Splitting Dataset

# In[13]:


x=df.drop(columns=['Profit'],axis=1)
y=df['Profit']


# In[23]:


x.head()


# In[24]:


y.head()


# In[16]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# # DecisionTreeRegressor:
# 

# In[17]:


from sklearn.tree import DecisionTreeRegressor
import math
from sklearn import metrics
dr=DecisionTreeRegressor(random_state=0)
dr.fit(x_train,y_train)
y_pred=dr.predict(x_test)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
mae = metrics.mean_absolute_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)
print( 'MSE: ', mse) 
print( 'RMSE: ', rmse) 
print( 'MAE: ', mae) 
print( 'R2 Score: ', r2) 


# # RandomForestRegressor:
# 

# In[18]:


from sklearn.ensemble import RandomForestRegressor
import math
from sklearn import metrics
rf=RandomForestRegressor(random_state=0)
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
mae = metrics.mean_absolute_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)
print( 'MSE: ', mse) 
print( 'RMSE: ', rmse) 
print( 'MAE: ', mae) 
print( 'R2 Score: ', r2) 


# # AdaBoostRegressor:
# 

# In[19]:


from sklearn.ensemble import AdaBoostRegressor
import math
from sklearn import metrics
ada=AdaBoostRegressor(random_state=0)
ada.fit(x_train,y_train)
y_pred=ada.predict(x_test)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
mae = metrics.mean_absolute_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)
print( 'MSE: ', mse) 
print( 'RMSE: ', rmse) 
print( 'MAE: ', mae) 
print( 'R2 Score: ', r2) 


# In[25]:


results = pd.DataFrame({
    'Model': ['Decision Tree Regressor','Random Forest Regressor','AdaBoost Regressor'],
    'R2 Score': [0.9764,0.9675,0.9292]})

result_df = results.sort_values(by='R2 Score', ascending=False)
result_df = result_df.set_index('R2 Score')
result_df


# In[ ]:


# DecisionTreeRegressor has the highest r2 score. Therefore we go with DecisionTreeRegressor


# # Prediction

# In[21]:


#Predicting the profit value of a company if the value of its R&D Spend, Administration Cost and Marketing Spend are given

# Define the value atributes to predict the profit
values = [162597.70,151377.59,443898.53]

# Reshape the input data to match the shape of the training data
values_reshaped = np.array(values).reshape(1, -1)

# Use the trained DecisionTreeRegressor model to predict the Profit
predicted = dr.predict(values_reshaped)

print('Profit of the company:',predicted)

