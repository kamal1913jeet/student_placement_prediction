# 🎓 Student Placement Prediction Using Machine Learning

# 📌 Project Overview
Student Placement Prediction is a Machine Learning project that predicts whether a student is likely to get placed or not based on academic performance, technical skills, internships, projects, and other factors.
The project uses a **Random Forest Classifier** model and provides an interactive web interface using **Streamlit**.

# 🎯 Objectives
- Predict student placement status using Machine Learning.
- Analyze important factors affecting placement.
- Provide an easy-to-use prediction system.
- Visualize model performance and important features.

# 🛠️ Technologies Used
- **Python**
- **Pandas** - Data processing
- **NumPy** - Numerical operations
- **Scikit-learn** - Machine Learning algorithms
- **Random Forest Classifier** - Prediction model
- **Streamlit** - Web application development
- **Matplotlib & Seaborn** - Data visualization
- **Joblib** - Model saving and loading

# 🤖 Machine Learning Model

### Algorithm Used:

**Random Forest Classifier**
The model is trained using student academic and skill-based information to predict placement status.

# Machine Learning Workflow:
Dataset
   ↓
Data Preprocessing
   ↓
Label Encoding
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Streamlit Deployment
   ↓
Placement Prediction

# 📊 Dataset Features
The dataset contains the following attributes:
- Branch
- College Tier
- CGPA
- Backlogs
- Coding Skills
- DSA Score
- Aptitude Score
- Communication Skills
- ML Knowledge
- System Design
- Internships
- Projects Count
- Certifications
- Hackathons
- Open Source Contributions
- Extracurricular Activities

# 📁 Project Structure
Student Placement Prediction
│
├── app.py
├── train_model.py
├── analysis.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── dataset
│   └── student_placement.csv
├── images
│   ├── about project.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── model_accuracy.png
│   ├── streamlit_output.png
│   └── student_prediction with model_analysis.png
└── model
    ├── placement_model.pkl
    ├── branch_encoder.pkl
    ├── tier_encoder.pkl
    ├── confusion_matrix.png
    └── feature_importance.png

## Project Screenshots

### About Project
![About Project](images/about project.png)

### Confusion Matrix
![Confusion Matrix](images/confusion_matrix.png)

### Feature Importance
![Feature Importance](images/feature_importance.png)

### Model Accuracy
![Model Accuracy](images/model_accuracy.png)

### Streamlit Output
![Streamlit Output](images/streamlit_output.png)

### Student Prediction with Model Analysis
![Student Prediction](images/student_prediction with model_analysis.png)

# 🚀 How to Run the Project

# 1. Clone the Repository
git clone <your-github-repository-link>
# 2. Install Required Libraries
pip install -r requirements.txt
# 3. Run the Streamlit Application
streamlit run app.py

# 📈 Model Evaluation
The project evaluates the model using:
- Accuracy Score
- Classification Report
- Confusion Matrix
Feature importance analysis is also performed to understand which factors influence placement prediction.

# 🖥️ Application Features
The Streamlit application provides:
✅ Student information input form  
✅ Placement prediction  
✅ Placement probability percentage  
✅ Confusion matrix visualization  
✅ Feature importance visualization  

# 🔮 Future Enhancements
- Add student login authentication.
- Add CSV upload for multiple student predictions.
- Deploy the application online.
- Compare multiple Machine Learning algorithms.
- Add real-time placement analytics dashboard.
# TECH TRAINER
**KAMALJEET KAUR**
DEVELOPER BY
Garima Singh

# 📜 Conclusion
This project demonstrates the application of Machine Learning in education analytics by predicting student placement outcomes. The Streamlit-based interface makes the prediction system simple, interactive, and user-friendly.
