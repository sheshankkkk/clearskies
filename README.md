# 🌤️ ClearSkies — Air Quality Prediction & Analysis (Machine Learning Project)

ClearSkies is a complete end-to-end **Air Quality Prediction System** built using  
Machine Learning, Data Preprocessing Pipelines, and an interactive **Streamlit Dashboard**.  
The system cleans the UCI Air Quality dataset, trains multiple ML models, selects the best one,  
and provides **real-time air quality category prediction** along with **hourly pollutant trends**  
and **best-time-of-day analysis**.

---

##  Features

###  **1. Complete Data Engineering Pipeline**
- Raw UCI dataset ingestion  
- Handles missing values and invalid sensor readings (`-200`)  
- Date & Time merging into `DateTime`  
- Feature extraction (Hour, Day, etc.)  
- Generated AQ categories using rule-based labelling

###  **2. Machine Learning Model Training**
- Models tested:  
  - Logistic Regression  
  - Decision Tree  
  - KNN  
  - Naive Bayes  
  - SVM  
- All evaluated on accuracy, precision, recall, F1-score  
- **DecisionTreeClassifier selected as best model (≈ 100% accuracy on test set)**  
- Final model exported as:  
  - `classification_model.joblib`  
  - `classification_scaler.pkl`  
  - `classification_features.json`

###  **3. Streamlit Dashboard (Professional UI)**
- Clean professional layout (no emojis, no sidebar)
- Dataset viewer labeled **Data**
- Select a row → Automatically fill prediction inputs  
- **Predict** button for controlled predictions  
- Hourly pollutant visualization for CO(GT)
- Best Time of the Day indicator (lowest pollution hour)
- Responsive, real-time analysis

---

## 📊 Air Quality Categories

| Category         | Meaning                          |
|------------------|----------------------------------|
| Good             | Safe                             |
| Moderate         | Acceptable but mild impact       |
| Unhealthy        | Harmful for sensitive groups     |
| Very Unhealthy   | Dangerous                        |
| Hazardous        | Severe health risk               |

---

## Tech Stack

### **Programming & Frameworks**
- Python 3.12
- Streamlit
- Scikit-Learn
- Pandas
- NumPy
- Plotly (Bar charts, trends)

### **Modeling & Evaluation**
- Train/Test Split (80/20)
- StandardScaler  
- DecisionTreeClassifier  
- Joblib for model persistence  

### **Development**
- VS Code  
- Git & GitHub  
- Virtual Environment (`venv`)

---

## 📂 Project Structure

clearskies/
│
├── app/
│ └── main.py # Streamlit Dashboard
│
├── clearskies/
│ ├── preprocess.py # Cleaning & preprocessing
│ ├── data_loader.py # Dataset ingestion
│ ├── classification_models.py# ML training & evaluation
│
├── data/
│ ├── raw/ # Raw UCI dataset
│ └── processed/ # Cleaned dataset
│
├── models/
│ ├── classification_model.joblib
│ ├── classification_scaler.pkl
│ ├── classification_features.json
│
├── rebuild_scaler.py # Rebuilds StandardScaler for 9 features
├── requirements.txt
└── README.md

**Results Summary**

Clean dataset size: ~9,357 rows
9 selected pollutant features
DecisionTree achieved near-perfect classification accuracy
Visualizations help users identify safe hours of the day

** Final Output**

ClearSkies gives:
Instant AQ Category Predictions
Hourly pollution trends
Best time of the day to be outdoors
A professional-grade ML dashboard
