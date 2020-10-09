import streamlit as st
import pandas as pd
import pickle 

st.write("""
# Predict Blood Donation for Future Expectancy
This app predicts the **Blood Donation** type!
""")

st.header('User Input Parameters')

def user_input_features():
    Recency = st.text_input('Recency - months since the last donation', 2)
    Frequency = st.text_input('Frequency - total number of donation', 40)
    Monetary = st.slider('Monetary - total blood donated in c.c.', 250, 12500, 1000)
    Time = st.slider('Time - months since the first donation', 0, 100, 20)
    data = {'Recency': Recency,
            'Frequency': Frequency,
            'Monetary': Monetary,
            'Time': Time}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)


load_log = pickle.load(open('blood_prediction_model_log.pkl','rb'))

if st.button('Predict'):
    y_predict = load_log.predict(df)
    y_test_predict_proba = load_log.predict_proba(df)
    st.subheader('Prediction Probability')
    predict_proba = pd.DataFrame(y_test_predict_proba,columns= ['Will Not Donate','Will Donate'])
    st.write(predict_proba)
    
    if  y_predict==0 :
        st.success('Will Not Donate')
    else :
        st.success('Will Donate')
    
   

