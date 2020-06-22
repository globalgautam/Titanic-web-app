#!/usr/bin/env python
# coding: utf-8

# In[1]:


#install/import packages libraries
from sklearn.metrics import make_scorer, accuracy_score # for accuracy
from sklearn.model_selection import train_test_split # for splitting the train-test data
from sklearn.ensemble import RandomForestClassifier # random forest model
from sklearn import preprocessing # EDA

import pandas as pd # EDA
import numpy as np # EDA
import sklearn # machine learning


# In[2]:


# code to see the output of multiple lines of codes
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'


# # random forest model on python using base data

# In[3]:


# read the titanic dataset
titanic_train = pd.read_csv('C:\\Users\\Gautam\\Documents\\R\\Datasets\\titanic\\train.csv')
titanic_test = pd.read_csv('C:\\Users\\Gautam\\Documents\\R\\Datasets\\titanic\\test.csv')

# impute the missing data
titanic_train.Embarked = titanic_train.Embarked.fillna('G')
titanic_test.Embarked = titanic_test.Embarked.fillna('G')
titanic_train.Cabin = titanic_train.Cabin.fillna('OTHR')
titanic_test.Cabin = titanic_test.Cabin.fillna('OTHR')
titanic_train.Age = titanic_train.Age.fillna(29.8)
titanic_test.Age = titanic_test.Age.fillna(29.8)

# dropping the missing value record from the dataset
titanic_test = titanic_test.dropna()


# In[4]:


# label endcoding for the object datatypes 
for col in ['Sex','Cabin','Embarked']:
    ltrain = preprocessing.LabelEncoder()
    ltrain = ltrain.fit(titanic_train[col])
    titanic_train[col] = ltrain.transform(titanic_train[col])
    
    ltest = preprocessing.LabelEncoder()
    ltest = ltest.fit(titanic_train[col])
    titanic_train[col] = ltest.transform(titanic_train[col])


# In[5]:


#independent_all = titanic_train.drop(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'SibSp','Parch', 'Ticket', 'Cabin', 'Embarked'],axis=1)
independent_all = titanic_train[['Age', 'Fare', 'Pclass','Sex','Embarked']]
dependent_all = titanic_train[['Survived']]

# split the dataset into train/test
x_train, x_test, y_train, y_test = train_test_split(independent_all, dependent_all, test_size = 0.3, random_state=100)

# Random forest model
rfc = RandomForestClassifier()

rfc.fit(x_train,y_train)

# prediction on train data
rfc_prediction_train = rfc.predict(x_train)
rfc_score_train=accuracy_score(y_train,rfc_prediction_train) # here y_test is actual value and rfc_prediction is predicted value

print('The score of this random forest on TRAIN is : ',round(rfc_score_train * 100,2), '%')

# prediction on test data
rfc_prediction_test = rfc.predict(x_test)
rfc_score_test=accuracy_score(y_test,rfc_prediction_test) # here y_test is actual value and rfc_prediction is predicted value

print('The score of this random forest on TEST is : ',round(rfc_score_test * 100,2), '%')


# # Streamlit

# In[6]:


import streamlit as st
import pickle

# save the model
filename = 'Streamlit_Titanic.sav'
pickle.dump(rfc, open(filename,'wb'))

# load the model
loaded_model = pickle.load(open(filename, 'rb'))
predictions = loaded_model.predict(x_test)

# create the header on the app page
st.markdown('# Titanic Survival')


# In[7]:


# create the dropdown menus for input fields
DropDown1 = pd.DataFrame({'Sex': ['Male', 'Female']})
DropDown2 = pd.DataFrame({'Passenger class': ['1', '2', '3'],
                          'Port of Embark': ['Cherbourg','Queenstown','Southampton']})


# In[8]:


# take user inputs
Age = st.number_input('Age')
Fare = st.number_input('Fare')

Pclass = st.selectbox('Passanger Class',DropDown2['Passenger class'].unique())
temp_Sex = st.selectbox('Gender',DropDown1['Sex'].unique())
temp_Embarked = st.selectbox('Port of Embark', DropDown2['Port of Embark'].unique())

if temp_Sex == 'Male':
    Sex = 1
else:
    Sex = 0

if temp_Embarked == 'Cherbourg':
    Embarked = 1
else:
    if temp_Embarked == 'Queenstown':
        Embarked = 2
    else:
        Embarked = 3


# In[9]:


# store the inputs
features = [Age, Fare, Pclass, Sex, Embarked]

# convert user inputs into an array for the model
int_features = [int(x) for x in features]
final_features = [np.array(int_features)]

if st.button('Predict'):
    prediction = loaded_model.predict(final_features)
    st.balloons()
    if round(prediction[0],2) == 0:
        st.success('Unfortunately this passenger could not survive.')
    else:
        st.success('Luckily this passenger survived.')

st.success(features)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




