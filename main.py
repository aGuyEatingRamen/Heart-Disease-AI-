#data orgin: https://www.kaggle.com/datasets/tejpal123/human-heart-disease-dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('heart.csv')

# Split features and target variable
X = data.drop('target', axis=1)
y = data['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression model (low accaracy)
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
#Gets accuracy for Logistic Regression model
lr_accuracy = accuracy_score(y_test, lr_pred)
print("Logistic Regression Accuracy:", lr_accuracy)
print(classification_report(y_test, lr_pred))

# Train SVM model (moderate accaracy)
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
#Gets accuracy for SVM model
svm_accuracy = accuracy_score(y_test, svm_pred)
print("\nSVM Accuracy:", svm_accuracy)
print(classification_report(y_test, svm_pred))

# Train Random Forest model (high accaracy)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
#Gets accuracy for Random Forest model
rf_accuracy = accuracy_score(y_test, rf_pred)
print("\nRandom Forest Accuracy:", rf_accuracy)
print(classification_report(y_test, rf_pred))


# Recives data from the user
while True:
    try:
        age = int(input("Enter the age of the patient: "))
        if age < 0 or age > 150:
            raise ValueError("Age must be between 0 and 150.")
        
        sex = int(input("Enter the sex of the patient (0 for female, 1 for male): "))
        if sex not in [0, 1]:
            raise ValueError("Sex must be 0 or 1.")
        
        cp = int(input("Enter chest pain type (0-3): "))
        if cp not in [0, 1, 2, 3]:
            raise ValueError("Chest pain type must be between 0 and 3.")
        
        trestbps = int(input("Enter resting blood pressure (in mm Hg): "))
        if trestbps < 0 or trestbps > 300:
            raise ValueError("Resting blood pressure must be between 0 and 300.")
        
        chol = int(input("Enter serum cholestoral in mg/dl: "))
        if chol < 0 or chol > 600:
            raise ValueError("Serum cholestoral must be between 0 and 600.")
        
        fbs = int(input("Enter fasting blood sugar > 120 mg/dl (1 for True, 0 for False): "))
        if fbs not in [0, 1]:
            raise ValueError("Fasting blood sugar must be 0 or 1.")
        
        restecg = int(input("Enter resting electrocardiographic results (0-2): "))
        if restecg not in [0, 1, 2]:
            raise ValueError("Resting electrocardiographic results must be between 0 and 2.")
        
        thalach = int(input("Enter maximum heart rate achieved: "))
        if thalach < 0 or thalach > 300:
            raise ValueError("Maximum heart rate achieved must be between 0 and 300.")
        
        exang = int(input("Enter exercise induced angina (1 for Yes, 0 for No): "))
        if exang not in [0, 1]:
            raise ValueError("Exercise induced angina must be 0 or 1.")
        
        oldpeak = float(input("Enter ST depression induced by exercise relative to rest: "))
        slope = int(input("Enter the slope of the peak exercise ST segment (0-2): "))
        if slope not in [0, 1, 2]:
            raise ValueError("Slope must be between 0 and 2.")
        
        ca = int(input("Enter number of major vessels (0-3) colored by fluoroscopy: "))
        if ca not in [0, 1, 2, 3]:
            raise ValueError("Number of major vessels must be between 0 and 3.")
        
        thal = int(input("Enter thalassemia (1-3): "))
        if thal not in [1, 2, 3]:
            raise ValueError("Thalassemia must be between 1 and 3.")
        
        model_choice = int(input("Please enter the number of the model you wish to use.\n\n\t\t*CAUTION*\n--When using Random Forest, be aware that the model may give slightly inaccurate results due to overfitting.--\n--When using this model, never rely on its results!--\n\n Logistic Regression (75%~)\t:\t1\nSVM(80%~)\t:\t2\nRandom Forest(97%~)\t:\t3\nNumber:"))
        if model_choice not in [1, 2, 3]:
            raise ValueError("Model choice must be 1, 2, or 3.")
        
        break  # If all inputs are valid, exit the loop
    except ValueError as ve:
        print("Invalid input:", ve)


# Manually standarizes the data from the user
user_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
X_train_mean = X_train.mean(axis=0)
X_train_std = X_train.std(axis=0)
user_data_scaled = (user_data - X_train_mean) / X_train_std


# Chooses model and predicts the values

if model_choice == 1:
    prediction = lr_model.predict(user_data)
    model_name = "Logistic Regression"
elif model_choice == 2:
    prediction = svm_model.predict(user_data)
    model_name = "SVM"
elif model_choice == 3:
    prediction = rf_model.predict(user_data)
    model_name = "Random Forest" 

print(f"\nPrediction using {model_name} (1 means postive, 0 means negative):", prediction)