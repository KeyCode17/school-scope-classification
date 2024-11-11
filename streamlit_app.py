# Importing necessary libraries

# - math: for mathematical operations
import math

# - os: set the directory location of the saved models
import os

# - sys: for system
import sys

# - rahdom: for random
import random

# - joblib: for saving and loading models
import joblib

# - squarify: for treemap
import squarify

# - numpy: for numerical operations
import numpy as np

# - dill: for serializing and deserializing Python objects
import dill as pickle

# - pandas: for data manipulation and analysis
import pandas as pd

# - seaborn: for data visualization
import seaborn as sns

# - matplotlib.pyplot: for data visualization
import matplotlib.pyplot as plt

# - matplotlib.lines: for data visualization
from matplotlib.lines import Line2D

# - sklearn.pipeline: for machine learning pipeline
from sklearn.pipeline import Pipeline

# - scipy.stats: for statistical tests
from scipy.stats import shapiro, spearmanr

# - sklearn.compose: for preprocessing data
from sklearn.compose import ColumnTransformer, make_column_transformer

# - sklearn.preprocessing: for preprocessing data
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder

# - tensorflow: for deep learning
import tensorflow as tf

# - tensorflow.keras.layers: for deep learning
from tensorflow.keras.layers import Dense, LeakyReLU

# - tensorflow.keras.models: for deep learning
from tensorflow.keras.models import Sequential, load_model

# - tensorflow.keras.callbacks: for deep learning
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# - sklearn.model_selection: for model selection and evaluation
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

# - streamlit: for web application
import streamlit as st
import streamlit_nested_layout
from streamlit_option_menu import option_menu
# End Of Imports

st.set_page_config(page_title='Student Performance', page_icon='📊')
st.title("School scope classification through the living environment around students")

# Setting the directory location
dirloc = os.path.dirname(os.path.abspath(__file__))

# Loading the dataset
data = pd.read_csv(f'{dirloc}/data/student-por.csv', delimiter=';')
data = pd.DataFrame(data)

# Renaming the columns
data.rename(columns={'sex': 'gender'}, inplace=True)

data = data.drop(columns=['failures', 'absences', 'G1', 'G2', 'G3'])

data_X = data.drop(columns='school').copy()
data_y = data['school'].copy()
data_y = pd.DataFrame(data_y, columns=['school'])

model_path = f'{dirloc}/model/model.keras'
model = load_model(model_path)

# Correlation Functions
with open(f'{dirloc}/function/all_functions.pkl', 'rb') as file:
    plot_combined_pie_charts, plot_combined_treemap_not2, plot_distribution, plot_boxplot, plot_normality_tests, heatmaps_spearman, get_top_correlations_with_column, plot_correlations_with_column = pickle.load(file)

# Normalize Data
pipeline = joblib.load(f'{dirloc}/function/pipeline.pkl')

with open(f'{dirloc}/function/transform_function.pkl', 'rb') as file:
    transform_and_sort, desired_prefix_order = pickle.load(file)
    data_transformed_sorted = transform_and_sort(data_X, pipeline, desired_prefix_order)

pipeline_y = joblib.load(f'{dirloc}/function/pipeline_y.pkl')

with open(f'{dirloc}/function/transform_function_y.pkl', 'rb') as file:
    transform_and_sort_y, desired_prefix_order_y = pickle.load(file)
    data_transformed_sorted_y = transform_and_sort_y(data_y, pipeline_y, desired_prefix_order_y)

scaler = joblib.load(f'{dirloc}/function/scaler.pkl')
data_transformed_scaled = scaler.transform(data_transformed_sorted)

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(data_X, data_y, test_size=0.2, random_state=15, stratify=data_transformed_sorted_y)


X_train, X_test, y_train, y_test = train_test_split(data_transformed_scaled, data_transformed_sorted_y, test_size=0.2, random_state=15, stratify=data_transformed_sorted_y)

# Sidebar
with st.sidebar:
    # Using st.form to wrap checkboxes
    with st.form(key='location_form'):
        st.subheader("Choose School")
        GP_selected = st.checkbox("Gabriel Pereira (GP)", value=True)
        MS_selected = st.checkbox("Mousinho da Silveira (MS)", value=True)

        # Submit button to reload the page
        apply_button = st.form_submit_button(label='Apply')

    # Check if at least one checkbox is selected
    if not (GP_selected or MS_selected):
        st.warning("Please select at least one location.")

# Check if at least one checkbox is selected
if not (GP_selected or MS_selected):
    st.error("Please select at least one location.")
    sys.exit()

# Filter data based on selected locations
if GP_selected and MS_selected:
    filtered_data = data
    name=None
elif GP_selected:
    filtered_data = data[data['school'] == "GP"]
    name="Data of Gabriel Pereira (GP)"
elif MS_selected:
    filtered_data = data[data['school'] == "MS"]
    name="Data of Mousinho da Silveira (MS)"
else:
    filtered_data = data
    name=None

# Defining the columns
# Filter columns with exactly two unique values
obj_columns = filtered_data.select_dtypes(include=['object']).columns

two_value_columns = [column for column in obj_columns if filtered_data[column].nunique() == 2]

nottwo_value_columns = [column for column in obj_columns if filtered_data[column].nunique() > 2]

int_columns = filtered_data.select_dtypes(include=['int64']).columns

int_columns_filtered1 = int_columns.drop(['age'])

int_columns_filtered2 = ['age']

@st.cache_data()
def show_data(url, idx_name, num=None):
    if num==None:
        df = pd.read_excel(url).set_index(idx_name)
    else:
        df = pd.read_excel(url).set_index(idx_name).head(num)
    return df

with st.expander('About this article'):
    st.markdown('**Abstract**')
    st.info("""
    This study investigates the classification of school scope based on students' surrounding life using Machine Learning (ML). The primary problem addressed is the influence of family background and social environment on determining school scope, with the objective to develop an effective predictive model. Employing a Deep Neural Network (DNN) algorithm within a supervised learning framework, data from the UCI Machine Learning Repository, encompassing performance metrics of 649 students from two Portuguese secondary schools, was analyzed. Spearman's rank correlation coefficient and the Shapiro-Wilk normality test were used to understand variable relationships. The study found that students' surrounding life significantly impacts school scope classification. The DNN model attained an accuracy of 0.83077, with an identical F1 score and precision of 0.83077 each, as well as a recall rate of 0.83077. Additionally, it recorded an AUC-ROC score of 0.85752. These findings suggest that ML models can effectively predict school scope, providing valuable insights for educators and policymakers to create conducive learning environments by considering students' backgrounds. This research contributes to developing targeted educational strategies and policies.
    """)    
    
    with st.expander('**Acknowledgments**'):
        st.info('''
    Praise to Allah SWT, the Creator of the universe, for His endless mercy and blessings. I also send my heartfelt regards to the Prophet Muhammad SAW, who has shown us the path of righteousness.

    I am deeply grateful to my parents for their unwavering support, prayers, and love throughout my journey. Their encouragement has been a guiding light in my life.

    I would also like to thank my advisor, Drs. Anis Zubair, M.Kom., for his invaluable guidance and constructive feedback during the preparation of this report. His support has been instrumental in helping me navigate the challenges I encountered.

    This report would not have been possible without the collaboration and assistance of everyone mentioned. I hope it contributes positively to the fields of knowledge and industry.
    ''') 
    
    st.markdown('**Notes**')
    st.markdown('Data sets:')
    doi = "[Student Performance and Living Environment](https://doi.org/10.24432/C5TG7T)"
    st.markdown(f'''{doi}''')

    deskripsi = show_data(f'{dirloc}/data/deskripsi_data.xlsx','Variable Name')

    with st.expander('Variable Table'):
        st.dataframe(deskripsi,use_container_width=True)

    with st.expander('Machine Learning Model'):
        with st.expander('Model With All Data', expanded=False):
            model_ml = st.columns((1,2))
            with model_ml[1]:
                st.image(f'{dirloc}/model/model_c.png', use_column_width=True)
            with model_ml[0]:
                st.image(f'{dirloc}/model/layer.png', use_column_width=True)
            with st.expander('Model Evaluation'):
                st.image(f'{dirloc}/model/evaluation_scores.png', use_column_width=True)

        with st.expander('Model only Top 5 Correlation', expanded=False):
            st.image(f'{dirloc}/model/corr-model.png', use_column_width=True)
            with st.expander('Model Evaluation'):
                st.image(f'{dirloc}/model/corr-evaluation_scores.png', use_column_width=True)
                
        with st.expander('Model WITHOUT Top 5 Correlation', expanded=False):
            st.image(f'{dirloc}/model/uncorr-model.png', use_column_width=True)
            with st.expander('Model Evaluation'):
                st.image(f'{dirloc}/model/uncorr-evaluation_scores.png', use_column_width=True)
        
        with st.expander('Model WITH Top 5 Correlation', expanded=False):
            st.image(f'{dirloc}/model/noncorr-model.png', use_column_width=True)
            with st.expander('Model Evaluation'):
                st.image(f'{dirloc}/model/noncorr-evaluation_scores.png', use_column_width=True)


    with st.expander('Libraries used'):
        cols1 = st.columns([1, 1])
        with cols1[0]:
            st.code('''
            - random: for random
            - squarify: for treemap
            - tensorflow: for deep learning
            - scipy.stats: for statistical tests
            - joblib: for saving and loading models
            - sklearn.compose: for preprocessing data
            - matplotlib.pyplot: for data visualizations
            - pandas: for data manipulation and analysis
            - streamlit: for web application development
            - seaborn: for statistical data visualization
            ''', language='markdown')
        with cols1[1]:
            st.code('''
            - sklearn.preprocessing: for scaling and encoding
            - sys: for system-related parameters and functions
            - os: set the directory location of the saved models
            - dill: for serializing and deserializing Python objects
            - sklearn.pipeline: for building machine learning pipelines
            - tensorflow.keras.callbacks: for callbacks during training
            - tensorflow.keras.models: for building neural network models
            - tensorflow.keras.layers: for building neural network layers
            - sklearn.model_selection: for model selection and evaluation
            - matplotlib.lines: for creating custom line elements in plots
            ''', language='markdown')

main_menu = option_menu(None, ["Analysis", "Machine Learning"], 
    icons=['bi-graph-up',  'bi-robot'], 
    menu_icon="cast", default_index=0, orientation="horizontal",
    styles={"nav-link": {"font-size": "20px", "text-align": "center", "margin":"0px", "--hover-color": "#eee"}})

if main_menu=='Analysis':
    if GP_selected and MS_selected:
        st.header('Data Schools', divider='rainbow')
    elif GP_selected:
        st.header('Data of School Gabriel Pereira', divider='red')
    elif MS_selected:
        st.header('Data of School Mousinho da Silveira', divider='red')
    else:
        st.header('Data', divider='rainbow')

    col = st.columns(4)
    col[0].metric(label="No. of Data", value=filtered_data.shape[0], delta="")
    col[1].metric(label="No. of Columns", value=filtered_data.shape[1], delta="")
    if GP_selected and MS_selected:
        col[2].metric(label="No. of Training samples", value=y_train.shape[0], delta="")
        col[3].metric(label="No. of Test samples", value=y_test.shape[0], delta="")
    elif GP_selected:
        col[2].metric(label="No. of Training samples", value=len(y_train[y_train['school'] == 1]), delta="")
        col[3].metric(label="No. of Test samples", value=len(y_test[y_test['school']  == 1]), delta="")
    elif MS_selected:
        col[2].metric(label="No. of Training samples", value=len(y_train[y_train['school']  == 0]), delta="")
        col[3].metric(label="No. of Test samples", value=len(y_test[y_test['school']  == 0]), delta="")    
    else:
        col[2].metric(label="No. of Training samples", value=y_train.shape[0], delta="")
        col[3].metric(label="No. of Test samples", value=y_test.shape[0], delta="")

    with st.expander('Initial dataset', expanded=True):
        st.dataframe(filtered_data, height=300,hide_index=True, use_container_width=True)

        with st.expander('Training samples', expanded=False):
            train_col = st.columns((3,1))
            with train_col[0]:
                st.markdown('**X**')
                st.dataframe(X_train_d, height=210, hide_index=True, use_container_width=True)
            with train_col[1]:
                st.markdown('**y**')
                st.dataframe(y_train_d, height=210, hide_index=True, use_container_width=True)

        with st.expander('Test samples', expanded=False):
            train_col = st.columns((3,1))
            with train_col[0]:
                st.markdown('**X**')
                st.dataframe(X_test_d, height=210, hide_index=True, use_container_width=True)
            with train_col[1]:
                st.markdown('**y**')
                st.dataframe(y_test_d, height=210, hide_index=True, use_container_width=True)

    selected = option_menu(name, ["Pie Chart", "Treemap", "Distributed", 'Boxplot', 'Shapiro', 'Spearman'], 
        icons=['bi-pie-chart-fill', 'bi-border-all', "bi-border-style", 'bi-align-center', 'bi-bar-chart-line-fill', 'bi-box-arrow-in-down-right'], 
        menu_icon="cast", default_index=0, orientation="horizontal",
        styles={"nav-link": {"font-size": "20px", "text-align": "center", "margin":"0px", "--hover-color": "#eee"}})

    if selected=='Pie Chart':
        plot_combined_pie_charts(two_value_columns)
        st.pyplot(plt.gcf())

    if selected=='Treemap':
        plot_combined_treemap_not2(nottwo_value_columns)
        st.pyplot(plt.gcf())

    if selected=='Distributed':
        plot_distribution(filtered_data, int_columns_filtered1)
        st.pyplot(plt.gcf())

    if selected=='Boxplot':
        plot_boxplot(filtered_data, int_columns_filtered2)
        st.pyplot(plt.gcf())

    if selected=='Shapiro':
        plot_normality_tests(filtered_data, int_columns_filtered1)
        st.pyplot(plt.gcf())

    if selected=='Spearman':
        correlation = filtered_data[int_columns_filtered1].copy()
        correlation['school'] = data_transformed_sorted_y
        int_columns_filtered1 = list(int_columns_filtered1)
        int_columns_filtered1.append('school')

        spearman_corr = heatmaps_spearman(correlation, int_columns_filtered1, 'Spearman Correlation');
        st.pyplot(plt.gcf())
        top_n = st.selectbox("Select number of top correlations", options=[5, 10], index=0)
        top_spearman = get_top_correlations_with_column(spearman_corr, 'school', top_n=top_n)
        plot_correlations_with_column(correlations=top_spearman,
        target_column='school',
        title=f"Top {top_n} Spearman Correlations with 'school'")
        st.pyplot(plt.gcf())

if main_menu=='Machine Learning':
    with st.form(key='my_form'):
        st.subheader("Input Data Manual")

        st.markdown(" ")
        form1 = st.columns((1,1,1,2))
        with form1[0]:
            gender = st.selectbox("Gender", ['Male', 'Female'], index=0)
        with form1[1]:
            age = st.number_input("Age", min_value=15, max_value=22, step=1, value=15)
        with form1[2]:
            address = st.selectbox("Address", ['Urban', 'Rural'], index=0)
        with form1[3]:
            famsize = st.selectbox(f"Family Size", ['Less or Equal to 3', 'Greater than 3'], index=0)
        st.markdown(" ")


        form2 = st.columns(3)
        with form2[0]:
            pstatus = st.selectbox("Parent Status", ['Living Together', 'Apart'], index=0)
        with form2[1]:
            medu = st.selectbox("Mother Education", ['None', 'Primary Education (4th Grade)', '5th to 9th Grade',  'Secondary Education', 'Higher Education'], index=0)
        with form2[2]:
            fedu = st.selectbox("Father Education", ['None', 'Primary Education (4th Grade)', '5th to 9th Grade',  'Secondary Education', 'Higher Education'], index=0)
        st.markdown(" ")
        
        
        form3 = st.columns(3)
        with form3[0]:
            mjob = st.selectbox("Mother Job", ['Teacher', 'Health Care', 'Civilian Services', 'At Home', 'Other'], index=0)
        with form3[1]:
            fjob = st.selectbox("Father Job", ['Teacher', 'Health Care', 'Civilian Services', 'At Home', 'Other'], index=0)
        with form3[2]:
            reason = st.selectbox("Reason to Choose this School", ['Close to Home', 'School Reputation', 'Course Preference', 'Other'], index=0)
        st.markdown(" ")
        
        
        form4 = st.columns(3)
        with form4[0]:
            guardian = st.selectbox("Guardian", ['Mother', 'Father', 'Other'], index=0)
        with form4[1]:
            traveltime = st.selectbox("Travel Time to School", ['<15 Min.', '15 to 30 Min.', '30 Min.', '1 Hour'], index=0)
        with form4[2]:
            studytime = st.selectbox("Weekly Study Time", ['<2 Hours', '2 to 5 Hours', '5 to 10 Hours', '>10 Hours'], index=0)
        st.markdown(" ")
        
        
        form5 = st.columns(4)
        with form5[0]:
            schoolsup = st.selectbox("Extra Educational", ['Yes', 'No'], index=0)
        with form5[1]:
            famsup = st.selectbox("Family Support", ['Yes', 'No'], index=0)
        with form5[2]:
            paid = st.selectbox("Extra Paid Course", ['Yes', 'No'], index=0)
        with form5[3]:
            activities = st.selectbox("Extra Curricular", ['Yes', 'No'], index=0)
        st.markdown(" ")
        
        
        form6 = st.columns(4)
        with form6[0]:
            nursery = st.selectbox("Nursery School", ['Yes', 'No'], index=0)
        with form6[1]:
            higher = st.selectbox("Plan Higher Education", ['Yes', 'No'], index=0)
        with form6[2]:
            internet = st.selectbox("Internet at Home", ['Yes', 'No'], index=0)
        with form6[3]:
            romantic = st.selectbox("In Relationship", ['Yes', 'No'], index=0)
        st.markdown(" ")
        
        
        form7 = st.columns(3)
        with form7[0]:
            famrel = st.selectbox("Family Relationships", ['Very Bad', 'Bad', 'Normal', 'Great', 'Excellent'], index=2)
        with form7[1]:
            freetime = st.selectbox("Free Time After School", ['Very Low', 'Low', 'Normal', 'High', 'Very High'], index=2)
        with form7[2]:
            goout = st.selectbox("Hangout with Friends", ['Very Low', 'Low', 'Normal', 'High', 'Very High'], index=2)
        st.markdown(" ")
        
        
        form8 = st.columns(3)
        with form8[0]:
            dalc = st.selectbox("Workday Alcohol Consumption", ['None', 'Very Low', 'Low', 'Normal', 'High', 'Very High'], index=3)
        with form8[1]:
            walc = st.selectbox("Weekend Alcohol Consumption", ['None', 'Very Low', 'Low', 'Normal', 'High', 'Very High'], index=3)
        with form8[2]:
            health = st.selectbox("Current Health Status", ['Very Bad', 'Bad', 'Normal', 'Great', 'Excellent'], index=2)
        st.markdown(" ")
        
        st.text("")
        st.subheader("Or Upload Your Excel File")
        st.markdown("[Template for Excel](https://github.com/KeyCode17/student-performance-classification/raw/master/excel_with_dropdowns_and_data.xlsx)")
        # File uploader for batch processing
        uploaded_file = st.file_uploader("Upload Excel file for batch processing", type=["xlsx", "xls"])

        # Submit button
        submitted = st.form_submit_button(label='Apply')

    if submitted:
        if uploaded_file is not None:
            batch_data = pd.read_excel(uploaded_file)
            predictions = []
            for index, row in batch_data.iterrows():
                try:
                    user_input = [
                        row['Gender'],
                        row['Age'],
                        row['Address'],
                        row['Family Size'],
                        row['Parent Status'],
                        row['Mother Education'],
                        row['Father Education'],
                        row['Mother Job'],
                        row['Father Job'],
                        row['Reason to Choose this School'],
                        row['Guardian'],
                        row['Travel Time to School'],
                        row['Weekly Study Time'],
                        row['Extra Educational Support'],
                        row['Family Support'],
                        row['Extra Paid Course'],
                        row['Extra Curricular Activities'],
                        row['Nursery School'],
                        row['Plan Higher Education'],
                        row['Internet at Home'],
                        row['In Relationship'],
                        row['Family Relationships'],
                        row['Free Time After School'],
                        row['Hangout with Friends'],
                        row['Workday Alcohol Consumption'],
                        row['Weekend Alcohol Consumption'],
                        row['Current Health Status']
                    ]

                    # Mapping for categorical variables
                    gender_mapping = {'Male': 'M', 'Female': 'F'}
                    address_mapping = {'Urban': 'U', 'Rural': 'R'}
                    famsize_mapping = {'Less or Equal to 3': 'LE3', 'Greater than 3': 'GT3'}
                    pstatus_mapping = {'Living Together': 'T', 'Apart': 'A'}
                    education_mapping = {
                        'None': 0, 
                        'Primary Education (4th Grade)': 1, 
                        '5th to 9th Grade': 2, 
                        'Secondary Education': 3, 
                        'Higher Education': 4
                    }
                    job_mapping = {
                        'Teacher': 'teacher', 
                        'Health Care': 'health', 
                        'Civilian Services': 'services', 
                        'At Home': 'at_home', 
                        'Other': 'other'
                    }
                    reason_mapping = {
                        'Close to Home': 'home', 
                        'School Reputation': 'reputation', 
                        'Course Preference': 'course', 
                        'Other': 'other'
                    }
                    guardian_mapping = {'Mother': 'mother', 'Father': 'father', 'Other': 'other'}
                    traveltime_mapping = {'<15 Min.': 1, '15 to 30 Min.': 2, '30 Min.': 3, '1 Hour': 4}
                    studytime_mapping = {'<2 Hours': 1, '2 to 5 Hours': 2, '5 to 10 Hours': 3, '>10 Hours': 4}
                    binary_mapping = {'Yes': 'yes', 'No': 'no'}
                    rating_mapping = {'Very Bad': 1, 'Bad': 2, 'Normal': 3, 'Great': 4, 'Excellent': 5}
                    freetime_mapping = {'Very Low': 1, 'Low': 2, 'Normal': 3, 'High': 4, 'Very High': 5}
                    consumption_mapping = {'None': 0, 'Very Low': 1, 'Low': 2, 'Normal': 3, 'High': 4, 'Very High': 5}

                    # Apply mappings
                    user_input[0] = gender_mapping.get(user_input[0], user_input[0])
                    user_input[2] = address_mapping.get(user_input[2], user_input[2])
                    user_input[3] = famsize_mapping.get(user_input[3], user_input[3])
                    user_input[4] = pstatus_mapping.get(user_input[4], user_input[4])
                    user_input[5] = education_mapping.get(user_input[5], user_input[5])
                    user_input[6] = education_mapping.get(user_input[6], user_input[6])
                    user_input[7] = job_mapping.get(user_input[7], user_input[7])
                    user_input[8] = job_mapping.get(user_input[8], user_input[8])
                    user_input[9] = reason_mapping.get(user_input[9], user_input[9])
                    user_input[10] = guardian_mapping.get(user_input[10], user_input[10])
                    user_input[11] = traveltime_mapping.get(user_input[11], user_input[11])
                    user_input[12] = studytime_mapping.get(user_input[12], user_input[12])

                    user_input[13] = binary_mapping.get(user_input[13], user_input[13])
                    user_input[14] = binary_mapping.get(user_input[14], user_input[14])
                    user_input[15] = binary_mapping.get(user_input[15], user_input[15])
                    user_input[16] = binary_mapping.get(user_input[16], user_input[16])
                    user_input[17] = binary_mapping.get(user_input[17], user_input[17])
                    user_input[18] = binary_mapping.get(user_input[18], user_input[18])
                    user_input[19] = binary_mapping.get(user_input[19], user_input[19])
                    user_input[20] = binary_mapping.get(user_input[20], user_input[20])

                    user_input[21] = rating_mapping.get(user_input[21], user_input[21])

                    user_input[22] = freetime_mapping.get(user_input[22], user_input[22])
                    user_input[23] = freetime_mapping.get(user_input[23], user_input[23])

                    user_input[24] = consumption_mapping.get(user_input[24], user_input[24])
                    user_input[25] = consumption_mapping.get(user_input[25], user_input[25])

                    user_input[26] = rating_mapping.get(user_input[26], user_input[26])

                    # Transform and scale the input data
                    user_df = pd.DataFrame([user_input], columns=[
                        'gender', 'age', 'address', 'famsize', 'Pstatus', 'Medu',
                        'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
                        'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
                        'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
                        'Walc', 'health'
                    ])
                    user_df_transformed = transform_and_sort(user_df, pipeline, desired_prefix_order)
                    user_df_scaled = scaler.transform(user_df_transformed)

                    # Make prediction
                    prediction = model.predict(user_df_scaled)[0]
                    
                    # Apply threshold of 0.5 for binary prediction
                    threshold = 0.5
                    binary_predictions = (prediction >= threshold).astype(float)

                    predictions.append("MS" if binary_predictions == 0 else "GP")
                except KeyError as e:
                    st.error(f"Error processing row {index + 2}: {e}. Make sure all columns are correct.")
                    sys.exit()
    
            batch_data.insert(0, 'Prediction Result', predictions)
            st.header("Batch Prediction Result")
            st.dataframe(batch_data)
        else:
            user_input = [
                gender,
                age,
                address,
                famsize,
                pstatus,
                medu,
                fedu,
                mjob,
                fjob,
                reason,
                guardian,
                traveltime,
                studytime,
                schoolsup,
                famsup,
                paid,
                activities,
                nursery,
                higher,
                internet,
                romantic,
                famrel,
                freetime,
                goout,
                dalc,
                walc,
                health
            ]

            # Mapping for categorical variables can be added here
            gender_mapping = {'Male': 'M', 'Female': 'F'}
            address_mapping = {'Urban': 'U', 'Rural': 'R'}
            famsize_mapping = {'Less or Equal to 3': 'LE3', 'Greater than 3': 'GT3'}
            pstatus_mapping = {'Living Together': 'T', 'Apart': 'A'}
            education_mapping = {
                'None': 0, 
                'Primary Education (4th Grade)': 1, 
                '5th to 9th Grade': 2, 
                'Secondary Education': 3, 
                'Higher Education': 4
            }
            job_mapping = {
                'Teacher': 'teacher', 
                'Health Care': 'health', 
                'Civilian Services': 'services', 
                'At Home': 'at_home', 
                'Other': 'other'
            }
            reason_mapping = {
                'Close to Home': 'home', 
                'School Reputation': 'reputation', 
                'Course Preference': 'course', 
                'Other': 'other'
            }
            guardian_mapping = {'Mother': 'mother', 'Father': 'father', 'Other': 'other'}
            traveltime_mapping = {'<15 Min.': 1, '15 to 30 Min.': 2, '30 Min.': 3, '1 Hour': 4}
            studytime_mapping = {'<2 Hours': 1, '2 to 5 Hours': 2, '5 to 10 Hours': 3, '>10 Hours': 4}
            binary_mapping = {'Yes': 'yes', 'No': 'no'}
            rating_mapping = {'Very Bad': 1, 'Bad': 2, 'Normal': 3, 'Great': 4, 'Excellent': 5}
            freetime_mapping = {'Very Low': 1, 'Low': 2, 'Normal': 3, 'High': 4, 'Very High': 5}
            consumption_mapping = {'None': 0, 'Very Low': 1, 'Low': 2, 'Normal': 3, 'High': 4, 'Very High': 5}

            user_input[0] = gender_mapping[user_input[0]]
            user_input[2] = address_mapping[user_input[2]]
            user_input[3] = famsize_mapping[user_input[3]]
            user_input[4] = pstatus_mapping[user_input[4]]
            user_input[5] = education_mapping[user_input[5]]
            user_input[6] = education_mapping[user_input[6]]
            user_input[7] = job_mapping[user_input[7]]
            user_input[8] = job_mapping[user_input[8]]
            user_input[9] = reason_mapping[user_input[9]]
            user_input[10] = guardian_mapping[user_input[10]]
            user_input[11] = traveltime_mapping[user_input[11]]
            user_input[12] = studytime_mapping[user_input[12]]

            user_input[13] = binary_mapping[user_input[13]]
            user_input[14] = binary_mapping[user_input[14]]
            user_input[15] = binary_mapping[user_input[15]]
            user_input[16] = binary_mapping[user_input[16]]
            user_input[17] = binary_mapping[user_input[17]]
            user_input[18] = binary_mapping[user_input[18]]
            user_input[19] = binary_mapping[user_input[19]]
            user_input[20] = binary_mapping[user_input[20]]

            user_input[21] = rating_mapping[user_input[21]]

            user_input[22] = freetime_mapping[user_input[22]]
            user_input[23] = freetime_mapping[user_input[23]]

            user_input[24] = consumption_mapping[user_input[24]]
            user_input[25] = consumption_mapping[user_input[25]]

            user_input[26] = rating_mapping[user_input[26]]
            
            # Transform and scale the input data
            user_df = pd.DataFrame([user_input], columns=[
                        'gender', 'age', 'address', 'famsize', 'Pstatus', 'Medu',
                        'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
                        'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
                        'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
                        'Walc', 'health'
                    ])
            # st.write(user_df)  # Display the DataFrame

            user_df_transformed = transform_and_sort(user_df, pipeline, desired_prefix_order)
            user_df_scaled = scaler.transform(user_df_transformed)

            # Make prediction
            prediction = model.predict(user_df_scaled)

            # Apply threshold of 0.5 for binary prediction
            threshold = 0.5
            binary_predictions = (prediction >= threshold).astype(float)

            st.header("Prediction Result")
            if binary_predictions == 0:
                st.success("The model predicts 'MS'.")
            elif binary_predictions == 1:
                st.success("The model predicts 'GP'.")
            else:
                st.error("The model predicts 'Fail'.")
