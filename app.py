import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt 
import xgboost as xgb
import shap 
import pickle 
import joblib

model = joblib.load('optimal_model.pkl')
scaler = pickle.load(open('scaler.pkl','rb'))
label_encoders = pickle.load(open('encoders.pkl','rb'))


st.set_page_config(page_title='Employee Salary Prediction App',page_icon='💰',layout='wide')
st.title('💼 Employee Salary Prediction System')
st.caption('Powered by XGBoost , SHAP and Streamlit - Predicts if an employee earns more than 50k or not.')

page = st.sidebar.radio("Navigate",['🔍 Single Prediction','📂 Batch Prediction','📈 Model Explanation'])

education_map = {
    'Preschool':1,'1st-4th':2,'5th-6th': 3,'7th-8th':4,'9th':5,
    '10th':6,'11th':7,'12th': 8,'HS-grad':9,'Some-college':10,
    'Assoc-voc':11,'Assoc-acdm':12,'Bachelors':13,'Masters':14,
    'Prof-school':15,'Doctorate':16
}

options = {
    "workclass":list(label_encoders['workclass'].classes_),
    "marital-status":list(label_encoders['marital-status'].classes_),
    "occupation":list(label_encoders['occupation'].classes_),
    "relationship":list(label_encoders['relationship'].classes_),
    "race":list(label_encoders['race'].classes_),
    "gender":list(label_encoders['gender'].classes_),
    "native-country":list(label_encoders['native-country'].classes_),
    "education":list(education_map.keys())
}

def preprocess_input(data):
    data['workclass'] =label_encoders['workclass'].transform(data['workclass'])
    data['marital-status'] =label_encoders['marital-status'].transform(data['marital-status'])
    data['occupation'] =label_encoders['occupation'].transform(data['occupation'])
    data['relationship'] = label_encoders['relationship'].transform(data['relationship'])
    data['race'] =label_encoders['race'].transform(data['race'])
    data['gender'] =label_encoders['gender'].transform(data['gender'])
    data['native-country'] =label_encoders['native-country'].transform(data['native-country'])
    return scaler.transform(data)

if page =='🔍 Single Prediction':
    st.subheader('👤 Input Employee Details')
    col1,col2,col3 = st.columns(3)
    with col1:
        age=st.slider('Age',17,65,30)
        workclass=st.selectbox('Workclass',options['workclass'])
        occupation=st.selectbox('Occupation',options['occupation'])
        capital_gain=st.slider('Capital Gain',0,100000,0,step=500)
    with col2:
        education=st.selectbox('Education',options['education'])
        marital_status=st.selectbox('Marital Status',options['marital-status']) 
        capital_loss=st.slider('Capital Loss',0,5000,0,step=100)
        hours_per_week=st.slider('Hours Per Week',1,100,40)
    with col3:
        experience=st.slider('Years of Experience',0,50,5)
        relationship=st.selectbox('Relationship',options['relationship'])
        race=st.selectbox('Race',options['race'])
        gender=st.radio('Gender',options['gender'],horizontal=True)
        native_country=st.selectbox('Native Country',options['native-country'])
    if st.button('Predict'):
        with st.spinner('Predicting...'):
            education_num = education_map[education]
            input_data =pd.DataFrame([[
                age, experience, workclass, education_num,
                marital_status, occupation, relationship,
                race, gender, capital_gain, capital_loss,
                hours_per_week, native_country
            ]], columns=[
                'age', 'experience', 'workclass', 'educational-num',
                'marital-status', 'occupation', 'relationship', 'race',
                'gender', 'capital-gain', 'capital-loss',
                'hours-per-week', 'native-country'
            ])
            X = preprocess_input(input_data.copy())
            pred = model.predict(X)[0]
            prob = model.predict_proba(X)[0][1]
            st.markdown('---------')
            if prob >= 0.5:
               pred_class = '>50K'
               confidence = prob
            else:
               pred_class = '<=50K'
               confidence = 1 - prob
            st.markdown(f"👔 The employee is likely to earn **{pred_class}** with a probability of **{confidence:.2f}**")
            # SHAP Explanation 
            explainer = shap.Explainer(model)
            shap_values = explainer(X)
            st.subheader('🔍 Why this Prediction ??')
            st.markdown('SHAP (SHapley Additive exPlanations) values explain the output of machine learning models by quantifying the contribution of each feature to the prediction.')
            st.markdown('The higher the SHAP value, the more the feature contributes to the prediction.')
            st.markdown('The SHAP values are calculated based on the model\'s predictions and the input features.')
            fig,ax = plt.subplots()
            shap.plots.waterfall(shap_values[0],max_display=10,show=False)
            st.pyplot(fig)
elif page== "📂 Batch Prediction":
    st.subheader('📊 Batch Prediction')
    st.markdown('Upload a CSV file with employee details to predict salaries for multiple employees.')
    uploaded_file = st.file_uploader("Choose a CSV file",type="csv")
    if uploaded_file is not None:
        try :
            df = pd.read_csv(uploaded_file)
            expected_cols = ['age','experience','workclass','education-num','marital-status','occupation','relationship','race','gender','capital-gain','capital-loss','hours-per-week','native-country']
            if all(col in df.columns for col in expected_cols):
               st.success("✅ File Uploaded Successfully ,Preview Here")
               st.dataframe(df.head())
               df_encoded = df.copy()
               df_encoded = preprocess_input(df_encoded)
               preds = model.predict(df_encoded)
               df['Prediction']=preds
               df['Income']=df['Prediction'].map({0:'Less than 50k',1:'More than 50k'})
               st.markdown('### Predictions')
               st.dataframe(df[['age','occupation','hours-per-week','Income']])
               csv = df.to_csv(index=False).encode('utf-8')
               st.download_button("📥 Download Predictions",csv,"predictions.csv","text/csv")
            else :
               st.error("❌ The uploaded file does not contain the required columns. Please check the file format.")
        except Exception as e:
            st.error(f"❌ An error occurred while processing the file: {e}")
elif page == "📈 Model Explanation":
    st.subheader('Global Feature Importance with XGBoost')
    st.write('Feature importance shows how much each feature contributes to the model\'s predictions.')
    st.write('The higher the importance score, the more the feature contributes to the model\'s predictions.')
    fig,ax=plt.subplots(figsize=(10,6))
    xgb_ax = xgb.plot_importance(model,ax=ax,height=0.5,max_num_features=13)
    st.pyplot(fig)
    st.info('You can also explore the SHAP values for individual predictions in the Single Prediction section.')

