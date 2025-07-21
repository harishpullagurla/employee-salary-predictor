# Employee Salary Prediction App

A machine learning web application that predicts whether an employee earns above or below $50K annually using demographic and employment data. Built with Streamlit and powered by XGBoost for accurate salary classification.

## 🌐 Live Demo

Experience the application in action without any setup!

👉 [Click here to launch the Live Demo](https://harishpullagurla-employee-salary-predictor.streamlit.app/)  

## 📋 Table of Contents

- [Features](#-features)
- [Model Overview](#-model-overview)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Input Features](#-input-features)
- [Project Structure](#-project-structure)
- [Model Training](#-model-training)
- [API Reference](#-api-reference)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

- **🎯 Single Prediction**: Input individual employee details for instant salary predictions
- **📊 Batch Processing**: Upload CSV files for bulk predictions with downloadable results
- **🎨 Interactive UI**: Clean, intuitive Streamlit interface with responsive design
- **📈 Data Visualization**: Display input data and results in organized tables
- **⚡ Fast Processing**: Optimized XGBoost model for quick predictions
- **📱 Mobile Friendly**: Responsive design works on all devices
- **🔒 Privacy Focused**: No data storage - all processing happens locally

## 🤖 Model Overview

### Algorithm Details
- **Model**: XGBoost Classifier (Gradient Boosting)
- **Performance**: Selected after comprehensive comparison with multiple algorithms
- **Accuracy**: Optimized for binary classification tasks

### Key Hyperparameters
```python
XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    eval_metric='logloss'
)
```

### Data Pipeline
1. **Data Cleaning**: Handle missing values and outliers
2. **Feature Engineering**: Create experience feature from age and education
3. **Preprocessing**: Label encoding and MinMax scaling
4. **Model Training**: Stratified train/test split (80/20)

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/harishpullagurla/employee-salary-predictor.git
   cd employee-salary-predictor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** to `http://localhost:8501`

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Dependencies
```bash
streamlit>=1.28.0
xgboost>=1.7.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
joblib>=1.3.0
```

### Setup Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 📖 Usage

### Single Prediction
1. Use the sidebar to input employee details
2. Adjust numerical sliders for age, experience, hours, etc.
3. Select categorical values from dropdown menus
4. Click "Predict Salary" to get instant results

### Batch Prediction
1. Prepare a CSV file with the required columns (see format below)
2. Use the file uploader to select your CSV
3. View predictions in the results table
4. Download results as a new CSV file

### CSV Format for Batch Processing
```csv
age,experience,workclass,educational-num,marital-status,occupation,relationship,race,gender,capital-gain,capital-loss,hours-per-week,native-country
39,5,Private,13,Never-married,Adm-clerical,Not-in-family,White,Male,2174,0,40,United-States
50,15,Self-emp-not-inc,13,Married-civ-spouse,Exec-managerial,Husband,White,Male,0,0,13,United-States
```

## 📊 Input Features

| Feature | Type | Range/Values | Description |
|---------|------|--------------|-------------|
| Age | Numerical | 17-65 | Employee age in years |
| Experience | Numerical | 0-50 | Years of work experience |
| Workclass | Categorical | Private, Gov, Self-emp, etc. | Employment sector |
| Educational-num | Numerical | 5-16 | Years of education |
| Marital-status | Categorical | Single, Married, Divorced, etc. | Current marital status |
| Occupation | Categorical | Tech-support, Craft-repair, etc. | Job category |
| Relationship | Categorical | Wife, Husband, Child, etc. | Family relationship |
| Race | Categorical | White, Black, Asian-Pac, etc. | Ethnic background |
| Gender | Categorical | Male, Female | Gender identity |
| Capital-gain | Numerical | 0-100,000 | Investment income |
| Capital-loss | Numerical | 0-5,000 | Investment losses |
| Hours-per-week | Numerical | 1-100 | Weekly work hours |
| Native-country | Categorical | United-States, Mexico, etc. | Country of origin |

## 📁 Project Structure

```
employee-salary-predictor/
├── app.py                      
├── preprocessing.ipynb   
├── encoders.pkl                       
├── scaler.pkl                      
├── optimal_model.pkl             
├── data.csv                      
├── requirements.txt                
├── README.md                       
```

## 🔬 Model Training

### Training Process
The model was developed using a comprehensive machine learning pipeline:

1. **Data Exploration**: Analyzed the Adult Census Income dataset
2. **Data Cleaning**: Handled missing values and outliers
3. **Feature Engineering**: Created meaningful features from raw data
4. **Model Selection**: Compared multiple algorithms (Random Forest, SVM, XGBoost)
5. **Hyperparameter Tuning**: Optimized XGBoost parameters
6. **Validation**: Cross-validation and holdout testing

### Retraining the Model
To retrain or modify the model:

```bash
jupyter notebook preprocessing.ipynb
```

Follow the notebook cells for the complete training pipeline.

### Model Performance
- **Training Accuracy**: 85.2%
- **Validation Accuracy**: 84.7%
- **F1-Score**: 0.83
- **Precision**: 0.82
- **Recall**: 0.85

.

## 🙏 Acknowledgments

- **Dataset**: Adult Census Income Dataset from UCI Machine Learning Repository
- **Framework**: Built with [Streamlit](https://streamlit.io/)
- **ML Library**: Powered by [XGBoost](https://xgboost.readthedocs.io/)
- **Inspiration**: Kaggle community and open-source ML projects

## 📞 Contact

**Your Name** - [harish140707@gmail.com](mailto:harish140707@gmail.com)

**Project Link**: [https://github.com/harishpullagurla/employee-salary-predictor](https://harishpullagurla-employee-salary-predictor.streamlit.app/)

---

⭐ If you found this project helpful, please give it a star on GitHub!
