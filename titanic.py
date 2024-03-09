from sklearn.base import BaseEstimator,TransformerMixin
from PIL import Image

import plotly.express as px
import streamlit as st
import seaborn as sns
import pandas as pd
import warnings
import pickle
import time

warnings.filterwarnings(action='ignore')

df=sns.load_dataset(name='titanic')

df.survived=df.survived.map(arg={1:'Survived',0:'Died'})

class InitialPreprocessor(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        self.nominal_features=['sex','embarked','who','embark_town']
        
        self.boolean_features=['adult_male','alone']
        
        return self
    
    def transform(self,X,y=None):
        X[self.nominal_features]=X[self.nominal_features].applymap(func=lambda x:x.strip().capitalize(),na_action='ignore')
        X[self.boolean_features]=X[self.boolean_features].applymap(func=lambda x: int(x))
        return X
    
    
    
titanic_image=Image.open(fp='Titanic.jpg.webp')

with open(file='clf_model.pickle',mode='rb') as pickled_model:
    model=pickle.load(file=pickled_model)
    
interface=st.container()

sidebar=st.sidebar.container()

with interface:
    st.title(body='Titanic Survival Prediction')
    
    st.image(image=titanic_image)
    
    st.header(body='Project Description')
    
    st.text(body="""
    This is a machine learning project that uses a supervised binary classification 
    algorithm to predict whether or not a passenger would survive the titanic accident.
    The model has been built in Python programming language using Scikit-Learn library.
    The model uses Logistic Regression algorithm to make predictions.
    """)
    
    st.write(px.pie(data_frame=df,names='survived'))
    
    st.subheader(body='Input Features')
    
    st.markdown(body='***')
    
    sex,adult_male,alone=st.columns(spec=[1,1,1])
    
    with sex:
        sex=st.radio(label='Your Gender',options=['Male','Female'],horizontal=True)
        
    with adult_male:
        adult_male=st.radio(label='Are you adult?',options=[True,False],horizontal=True)
        
    with alone:
        alone=st.radio(label='Are you alone?',options=[True,False],horizontal=True)
        
    st.markdown('***')
    
    age=st.slider(label='Age',min_value=1,max_value=100,value=int(df.age.mean()))
    
    st.markdown(body='***')
    
    sibsp=st.slider(label='Number of Siblings & Spouse Aboard',min_value=int(df.sibsp.min()),max_value=20,value=3)
    
    st.markdown(body='***')
    
    parch=st.slider(label='Number of Parent & Children Aboard',min_value=int(df.parch.min()),max_value=20,value=5)
    
    st.markdown('***')
    
    fare=st.slider(label='Ticket Fare',min_value=int(df.fare.min()),max_value=int(df.fare.max()),value=int(df.fare.mean()))
    
    st.markdown('***')
    
    class_,who,embark_town=st.columns(spec=[1,1,1])
    
    with class_:
        class_=st.selectbox(label='Your social class',options=['First','Second','Third'])
        
    with who:
        who=st.selectbox(label='You are',options=['Man','Woman','Child'])
        
    with embark_town:
        embark_town=st.selectbox(label='Embark Town',options=['Southampton','Cherbourg','Queenstown'])
        
    class_dictionary={'First':1,'Second':2,'Third':3}
    
    embarked=embark_town[0]
    
    pclass=class_dictionary.get(class_)
    
    data_dictionary = {'pclass':pclass,
                       'sex':sex,
                       'age':age,
                       'sibsp':sibsp,
                       'parch':parch,
                       'fare':fare,
                       'embarked':embarked,
                       'class':class_,
                       'who':who,
                       'adult_male':adult_male,
                       'embark_town':embark_town,
                       'alone':alone}
        
        
    input_df=pd.DataFrame(data=data_dictionary,index=[0])
    
    
    st.subheader(body='Model Prediction')
    
    if st.button('Predict'):
        
        survival_probability=model.predict_proba(X=input_df).ravel()[1]
        
        with st.spinner(text='Sending input features to model...'):
            
            time.sleep(2)
            
            st.success('Your prediction is ready!')
            
            time.sleep(1)
            
            st.markdown(body=f'Model output: Your chance of survival is **{survival_probability:.0%}**')
            
with sidebar:

    st.title(body = 'Variable Dictionary')
    
    st.markdown(body = '- **sex** - This variable indicates the gender of a passenger as string')
    st.markdown(body = '- **age** - This variable indicates the age of a passenger as float')
    st.markdown(body = '- **sibsb** - This variable indicates the number of siblings and spouses aboard as integer')
    st.markdown(body = '- **parch** - This variable indicates the number of parents and children aboard as integer')
    st.markdown(body = '- **fare** - This variable indicates the ticket price for a particular passenger as float')
    st.markdown(body = '- **class** - This variable indicates the social class of a passenger as string')
    st.markdown(body = '- **who** - This variable indicates if a passenger is a man, woman or a child as string')
    st.markdown(body = '- **adult_male** - This variable indicates if a male passenger is an adult or not as boolean')
    st.markdown(body = '- **embark_town** - This variable indicates the embarkation port as string')
    st.markdown(body = '- **alone** - This variable indicates if a passenger was alone or not')
                        
                        
            
    
    
    
    
    
    
    
    
