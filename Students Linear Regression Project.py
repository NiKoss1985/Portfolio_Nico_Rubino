#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[32]:


results = pd.read_csv(r"C:\Users\Acer\Downloads\archive (2)\Student_Performance.csv")


# In[33]:


results.info()


# In[34]:


results.describe()


# In[36]:


#exploring first correlations between avg.score and other columns?
sns.set_palette('GnBu_d')
sns.set_style('whitegrid')
sns.jointplot(x='Hours Studied', y='Performance Index', data= results)


# In[37]:


sns.pairplot(results)


# In[38]:


#Exploratory Data to see in detail the Performance vs. Previous Score
sns.set_palette('GnBu_d')
sns.set_style('whitegrid')
sns.jointplot(x='Previous Scores', y='Performance Index', data=results)


# In[40]:


#Exploratory Data to see in detail the Performance vs. Previous Score
sns.set_palette('GnBu_d')
sns.set_style('whitegrid')
sns.jointplot(x='Hours Studied', y='Performance Index', data=results)


# In[46]:


#we look at a linear relation & plot the line colour white
sns.lmplot(x='Previous Scores', y='Performance Index', data=results, line_kws={'color': 'white'})


# In[48]:


#Training & Test 
from sklearn.model_selection import train_test_split
y = results['Performance Index']
X = results[['Previous Scores', 'Hours Studied', 'Sleep Hours', 'Sample Question Papers Practiced']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[49]:


#Training the model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)


# In[50]:


# The coefficients
print('Coefficients: \n', lm.coef_)


# In[53]:


predictions = lm.predict( X_test)


# In[54]:


plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# In[55]:


#Evaluating the model - MAE, MSE, and RMSE values are relatively low, which is a positive sign.
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[60]:


#Residuals
#Our model is pretty accurate. No outliers. 
sns.displot((y_test-predictions),bins=50);


# In[61]:


#Final Interpretation: Hours Studied is the winner! 
#Every h2.8 studied increases 1 unit of the performance index 
#that will have an impact on the overall slope of the LR function. 
coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coefficient']
coeffecients


# In[ ]:




