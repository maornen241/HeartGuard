import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, confusion_matrix
# import matplotlib.pyplot as plt

st.set_page_config(
    page_title="HeartGuard: Your Heart Health Ally!",
    page_icon="💙",
)


st.markdown("# HeartGuard: Your Heart Health Ally! 🌟💙")
st.markdown("<br>", unsafe_allow_html=True) #blank line

st.markdown("""Dear Doctors, welcome to HeartGuard, your ultimate heart health prediction tool! 🩺💙

In this user-friendly platform, you, our dedicated healthcare heroes, can effortlessly input various patient features using the sidebar on the left. The magic unfolds behind the scenes as our cutting-edge algorithm processes this data and provides a personalized prediction: the likelihood of a patient having a heart disease.

We're here to empower you with precision and efficiency, making heart health assessments quicker, more accurate, and seamlessly integrated into your daily workflow. Your commitment to patient care, coupled with our intuitive tool, aims to enhance early detection and improve overall patient outcomes.

Your dedication to the well-being of others is truly commendable, and we're here to support you every step of the way. Together, let's make strides in heart health and ensure a brighter, healthier future for all. 🩺💙""")

df = pd.read_csv('https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv')

X = df.iloc[:,:-1]  # train dataset
y = df.iloc[:, -1]  # target variable

lr_model = LogisticRegression()
lr_model.fit(X,y)

st.sidebar.markdown('# User Input Parameters')

# Get the user parameters, the test results of the patient and put them into df that we give the model to predict on.
def user_input_features():
    age = st.sidebar.slider('age', 29, 77, 50)
    sex = st.sidebar.slider('sex', 0, 1, 0)
    cp = st.sidebar.slider('cp', 0, 3, 1)
    trestbps = st.sidebar.slider('trestbps', 94, 200, 95)
    chol = st.sidebar.slider('chol', 126, 564, 250)
    fbs = st.sidebar.slider('fbs', 0, 1, 1)
    restecg = st.sidebar.slider('restecg', 0, 2, 1)
    thalach = st.sidebar.slider('thalach', 71, 202, 100)
    exang = st.sidebar.slider('exang', 0, 1, 1)
    oldpeak = st.sidebar.slider('oldpeak', 0.0, 6.2, 1.0)
    slope = st.sidebar.slider('slope', 0, 2, 1)
    ca = st.sidebar.slider('ca', 0, 4, 1)
    thal = st.sidebar.slider('thal', 0, 3, 1)
    data = {'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal}
    features = pd.DataFrame(data, index=[0])
    return features

user_df = user_input_features()

st.markdown("<br>", unsafe_allow_html=True)

#st.markdown("## User input:")
#st.write(user_df)

prediction_proba = lr_model.predict_proba(user_df)
prediction = lr_model.predict(user_df)
st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
   st.markdown('### Prediction Probabilities:')
   st.write(f"Probability of your patient being **sick: {prediction_proba[0,1]*100:.3f}%**")
   st.write(f"Probability of your patient being **healthy: {prediction_proba[0, 0]*100:.3f}%**")
#   st.markdown('##### 0 -> *healthy* | 1 -> *sick*')
#   st.write(prediction_proba)


with col2:
   st.markdown('### Recommendation:')

   if prediction_proba[0,1] < 0.4:
       st.markdown('No apparent heart disease. Encourage a healthy lifestyle. Regular check-ups.')
   elif prediction_proba[0,1] < 0.6:
       st.markdown('Prediction uncertain. Recommend further diagnostic tests.')
   else:
       st.markdown("Consider additional tests to confirm **heart disease diagnosis.** Monitor closely.")

# Features Explanation button, first click show explanation, second click hide explanation
if 'button_state' not in st.session_state:
    st.session_state.button_state = False

feat_expl_button = st.sidebar.button("Features Explanation")

if feat_expl_button:
    if not st.session_state.button_state:
        st.session_state.button_state = True
        st.sidebar.write("""
    **Age**: displays the age of the individual.
    
    **Sex**: displays the gender of the individual using the following format :\n
    1 = male\n
    0 = female
    
    **cp** - Chest-pain type: displays the type of chest-pain experienced by the individual using the following format :\n
    1 = typical angina\n
    2 = atypical angina\n
    3 = non — anginal pain\n
    4 = asymptotic
    
    **trestbps** - Resting Blood Pressure: displays the resting blood pressure value of an individual in mmHg (unit)
    
    **chol** - Serum Cholestrol: displays the serum cholesterol in mg/dl (unit)
    
    **fbs** - Fasting Blood Sugar: compares the fasting blood sugar value of an individual with 120mg/dl.
    If fasting blood sugar > 120mg/dl then :\n 
    1 (true)\n
    else : 0 (false)
    
    **restecg** - Resting ECG : displays resting electrocardiographic results:\n
    0 = normal\n
    1 = having ST-T wave abnormality\n
    2 = left ventricular hyperthrophy
    
    **thalach** - Max heart rate achieved : displays the max heart rate achieved by an individual.
    
    **exang** - Exercise induced angina :\n
    1 = yes\n
    0 = no
    
    **oldpeak** - ST depression induced by exercise relative to rest: displays the value which is an integer or float.
    
    **slope** - Peak exercise ST segment :\n
    1 = upsloping\n
    2 = flat\n
    3 = downsloping
    
    **ca** - Number of major vessels (0–3) colored by flourosopy : displays the value as integer or float.
    
    **Thal** : displays the thalassemia :\n
    3 = normal\n
    6 = fixed defect\n
    7 = reversible defect
    
    **target** - Diagnosis of heart disease : Displays whether the individual is suffering from heart disease or not :\n
    0 = absence\n
    1 = present.
        """)

    else:
        st.session_state.button_state = False



# st.markdown('## Data')
# st.write(df)
# data_statistic = df.describe()
# st.markdown("### Data Statistic")
# st.write(data_statistic)
# st.write("Accuracy on training set:", lr_model.score(X, y))