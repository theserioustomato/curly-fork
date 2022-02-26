#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import RandomForestClassifier


# In[2]:


df = pd.read_csv("CreditCardUpgrade(1).csv")


# In[6]:



X = df.loc[:,["Purchases", "SuppCard"]]
Y = df.loc[:,"Upgraded"]


# In[9]:


#X = df.loc[:,["Purchases", "SuppCard"]]
#Y = df.loc[:,["Upgraded"]]


# In[7]:


model = tree.DecisionTreeClassifier(max_depth=3)
model.fit(X, Y)
pred = model.predict(X)
cm = confusion_matrix(Y, pred)
print(cm)
accuracy = (cm[0,0] + cm[1,1])/sum(sum(cm))
print(accuracy)


# In[10]:


joblib.dump(model, "CCU_DT")


# In[14]:


model = linear_model.LogisticRegression()
model.fit(X, Y)
pred = model.predict(X)
cm = confusion_matrix(Y, pred)
print(cm)
accuracy = (cm[0,0] + cm[1,1])/sum(sum(cm))
print(accuracy)

joblib.dump(model, "CCU_Reg")


# In[15]:


model = MLPClassifier(solver="lbfgs", hidden_layer_sizes=(6,6))
model.fit(X, Y)
pred = model.predict(X)
cm = confusion_matrix(Y, pred)
print(cm)
accuracy = (cm[0,0] + cm[1,1])/sum(sum(cm))
print(accuracy)

joblib.dump(model, "CCU_NN")


# In[22]:


model = RandomForestClassifier()
model.fit(X, Y)
pred = model.predict(X)
cm = confusion_matrix(Y, pred)
print(cm)
accuracy = (cm[0,0] + cm[1,1])/sum(sum(cm))
print(accuracy)

joblib.dump(model, "CCU_RF")


# In[23]:


model = GBC()
model.fit(X, Y)
pred = model.predict(X)
cm = confusion_matrix(Y, pred)
print(cm)
accuracy = (cm[0,0] + cm[1,1])/sum(sum(cm))
print(accuracy)

joblib.dump(model, "CCU_GBC")


# In[ ]:




