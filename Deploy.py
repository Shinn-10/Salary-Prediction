import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings(action = 'ignore')

st.title('Salary Prediction')

st.header('_'*50)




# Model Info
st.header('Our Performance')

# Error rate
st.text(f'Mean absolute error : 5966.9708')

# Performance
st.text(f'Model\'s Performance on training set : 97.83 %')
st.text(f'Model\'s Performance on testing set : 96.38 %')

st.header('_'*50)


# UI

st.header('User Section')

st.text('We need a few of your personal information to estimate your salary.')

# Age
age = st.slider('How old are you?', 0, 70, 1)
age_f = float(age)

# Gender


col1, col2 = st.columns(2)

with col1:
    st.header('Gender')
    gender = st.selectbox(
        'Can you expose your gender pls : ',
        ('Male', 'Female'))

with col2:
    st.header('Senior or Junior')
    level = st.selectbox(
        'Are you a senior or junior at work : ',
        ('Senior', 'Junior'))
    
    
    if level == 'Senior':
        level = 0
    else:
        level = 1

        
st.subheader('_' * 50)

# Country
st.header('Country')
country = st.selectbox(
    'Which country are you currently working in : ',
    ('UK', 'USA', 'Canada', 'China', 'Australia'))

# Year of Experience

st.header('Experience')
experience = st.text_input('Input a number between 0 and 35 with one decimal : ', '1')

y_exp = float(experience)


with st.sidebar:
    st.subheader('Welcome to our project')
    
    race = st.selectbox(
     'Please let me know your Race : ',
     ('White', 'Hispanic', 'Asian', 'Korean', 'Chinese', 'Australian',
       'Welsh', 'African American', 'Mixed', 'Black'))

    st.text(f'Wow "{race}" are gorgeous I think.')  
    
    st.header('_' * 30)
    
    # Job
    job = st.selectbox(
        'What is your occupation : ',
        ('Software Engineer', 'Data Analyst', 'Manager', 'Sales Associate',
       'Director', 'Marketing Analyst', 'Product Manager',
       'Sales Manager', 'Marketing Coordinator', 'Scientist',
       'Software Developer', 'HR Manager', 'Financial Analyst',
       'Project Manager', 'Customer Service Rep', 'Operations Manager',
       'Marketing Manager', 'Engineer', 'Data Entry Clerk',
       'Sales Director', 'Business Analyst', 'VP of Operations',
       'IT Support', 'Recruiter', 'Financial Manager',
       'Social Media Specialist', 'Software Manager', 'Developer',
       'Consultant', 'Product Designer', 'CEO', 'Accountant',
       'Data Scientist', 'Marketing Specialist', 'Technical Writer',
       'HR Generalist', 'Project Engineer', 'Customer Success Rep',
       'Sales Executive', 'UX Designer', 'Operations Director',
       'Network Engineer', 'Administrative Assistant',
       'Strategy Consultant', 'Copywriter', 'Account Manager',
       'Director of Marketing', 'Help Desk Analyst',
       'Customer Service Manager', 'Business Intelligence Analyst',
       'Event Coordinator', 'VP of Finance', 'Graphic Designer',
       'UX Researcher', 'Social Media Manager', 'Director of Operations',
       'Digital Marketing Manager', 'IT Manager',
       'Customer Service Representative', 'Business Development Manager',
       'Web Developer', 'Research Director',
       'Technical Support Specialist', 'Creative Director',
       'Human Resources Director', 'Content Marketing Manager',
       'Technical Recruiter', 'Sales Representative',
       'Chief Technology Officer', 'Designer', 'Financial Advisor',
       'Principal Scientist', 'Supply Chain Manager',
       'Training Specialist', 'Research Scientist',
       'Public Relations Manager', 'Operations Analyst',
       'Product Marketing Manager', 'Project Coordinator',
       'Chief Data Officer', 'Digital Content Producer',
       'IT Support Specialist', 'Customer Success Manager',
       'Software Project Manager', 'Supply Chain Analyst',
       'Office Manager', 'Principal Engineer', 'Sales Operations Manager',
       'Web Designer', 'Director of Sales', 'Customer Support Specialist',
       'Director of Human Resources', 'Director of Product Management',
       'Human Resources Manager', 'Business Development Associate',
       'Researcher', 'HR Coordinator', 'Director of Finance',
       'Human Resources Coordinator', 'IT Project Manager',
       'Quality Assurance Analyst', 'Director of Sales and Marketing',
       'Account Executive', 'Director of Business Development',
       'Human Resources Specialist', 'Director of Human Capital',
       'Advertising Coordinator', 'Marketing Director', 'IT Consultant',
       'Business Operations Analyst', 'Product Development Manager',
       'Software Architect', 'HR Specialist', 'Data Engineer',
       'Operations Coordinator', 'Director of HR',
       'Director of Engineering', 'Software Engineer Manager',
       'Back end Developer', 'Full Stack Engineer', 'Front end Developer',
       'Front End Developer', 'Director of Data Science',
       'Juniour HR Generalist', 'Juniour HR Coordinator',
       'Digital Marketing Specialist', 'Receptionist', 'Social Media Man',
       'Delivery Driver'))
    
    st.text(f'{job} ? That\'s cool.')
    
    st.header('_' * 30)
    
    # Education Level
    
    e_level = st.slider('Describe your education level :', 0, 3, 1)
#     st.text(f'Eduction level : {e_level} and {type(e_level)}')
 


    
    
    
# Button 

st.header('_' * 50)

st.header('Prediction')

st.text('Confirm your information and estimate your salary.')

if st.button('Confirm and Predict'):
    
    user_df = pd.DataFrame({
        'Age': [age_f],
        'Gender': [gender],
        'Education Level': [e_level],
        'Job Title' : [job],
        'Years of Experience' : [y_exp],
        'Country' : [country],
        'Race' : [race],
        'Senior' : [level]
    })
    
    
    st.text('Here is your personal data')
    
    st.dataframe(user_df)
    
    df = pd.read_csv('Salary.csv')

    data = df.copy()
    
    

    # Split features and label
    X = data.drop(['Salary'], axis = 1)
    y = data.Salary

    # Select categorical columns to be encoded
    object_cols = [col for col in X.columns if X[col].dtypes == 'object']

    encoder = OrdinalEncoder()

    X[object_cols] = encoder.fit_transform(X[object_cols])
    
    user_df[object_cols] = encoder.transform(user_df[object_cols])

    # Breaking off validation set from training set
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 42)


    # Standardization
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_valid_scaled = scaler.transform(X_valid)
    
    user_scaled = scaler.transform(user_df)


    # Modeling
    model = XGBRegressor(n_estimators = 1000, n_jobs = 4, learning_rate = 0.01, random_state = 42)

    model.fit(X_train_scaled, y_train,
             early_stopping_rounds = 5,
             eval_set = [(X_valid_scaled, y_valid)])

    preds = model.predict(X_valid_scaled)

    score = mean_absolute_error(y_valid, preds)

   
   
    
    # user dataframe

    # st.text(f'{user_scaled}')
    
    your_salary = model.predict(user_scaled)
    
    st.text(f'Your Estimated Salary : {your_salary}')
    
    st.success('This is a success message!', icon = "✅")
        
    
        
st.header('_' * 50 )

st.header('Thank You for Using My App')
