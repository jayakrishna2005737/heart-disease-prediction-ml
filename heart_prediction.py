# Predict whether a patient has heart disease using clinical features
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
df=pd.read_csv("heart_disease_uci.csv")
print("First 5 Rows:")
print(df.head())
print("\nDataset Shape:")
print(df.shape)
print("This dataset contains patient medical records used for heart disease prediction.")
numeric_cols=['trestbps','chol','thalch','oldpeak','ca']
for col in numeric_cols:
    df[col]=df[col].fillna(df[col].median())
categorical_cols=['fbs','restecg','exang','slope','thal']
for col in categorical_cols:
    df[col]=df[col].fillna(df[col].mode()[0])
print(df.isnull().sum())
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True),annot=True,cmap='coolwarm',fmt='.2f')
plt.title('Correlation Heatmap-heart_disease_uci.csv')
plt.show()
df['num']=df['num'].apply(lambda x: 1 if x > 0 else 0)
X=df.drop(["num","id"],axis=1)
y=df["num"]
X=pd.get_dummies(X,drop_first=True)
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)
print("X_train shape:",X_train.shape)
print("X_test shape:",X_test.shape)
log_model=Pipeline([
    ("scaler",StandardScaler()),
    ("model",LogisticRegression(max_iter=1000))
])
log_model.fit(X_train, y_train)
log_pred=log_model.predict(X_test)
log_accuracy=accuracy_score(y_test,log_pred)
print("\n===== Logistic Regression =====")
print("Accuracy:",log_accuracy)
print("Confusion Matrix:\n",confusion_matrix(y_test,log_pred))
print("Classification Report:\n",classification_report(y_test,log_pred))
rf_model=Pipeline([
    ("model",RandomForestClassifier(n_estimators=100,random_state=42))
])
rf_model.fit(X_train,y_train)
rf_pred=rf_model.predict(X_test)
rf_accuracy=accuracy_score(y_test,rf_pred)
print("\n===== Random Forest =====")
print("Accuracy:",rf_accuracy)
print("Confusion Matrix:\n",confusion_matrix(y_test,rf_pred))
print("Classification Report:\n",classification_report(y_test,rf_pred))
print("\n===== Model Comparison =====")
print(f"Logistic Regression Accuracy:{log_accuracy:.4f}")
print(f"Random Forest Accuracy:{rf_accuracy:.4f}")
if rf_accuracy > log_accuracy:
    print("Random Forest performed better!")
else:
    print("Logistic Regression performed better!")
print("\nInsight: Random Forest generally performs better as it captures non-linear relationships and feature interactions more effectively than Logistic Regression.")
sample=X_test.iloc[0:1]
prediction=rf_model.predict(sample)
actual=y_test.iloc[0]
print("\n===== Sample Prediction =====")
print("Predicted:","Heart Disease" if prediction[0]==1 else "No Heart Disease")
print("Actual:","Heart Disease" if actual==1 else "No Heart Disease")
print("\nFinal Conclusion: Machine Learning models successfully predict heart disease based on clinical features.")