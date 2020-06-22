#!/usr/bin/env python
# coding: utf-8

# In[10]:


import streamlit as st
import pickle
import pandas as pd
import numpy as np

# save the model
filename = 'Streamlit_Titanic.sav'

# load the model
loaded_model = pickle.load(open(filename, 'rb'))

# create the header on the app page
st.markdown('# Titanic Survival')


# In[11]:


# create the dropdown menus for input fields
DropDown1 = pd.DataFrame({'Sex': ['Male', 'Female']})
DropDown2 = pd.DataFrame({'Passenger class': ['1', '2', '3'],
                          'Port of Embark': ['Cherbourg','Queenstown','Southampton']})


# In[12]:


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


# In[13]:


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




