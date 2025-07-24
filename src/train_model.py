import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
DATA_PATH = 'Multiclass_Diabetes_Dataset.csv'
df = pd.read_csv(DATA_PATH)

# Features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Export model
joblib.dump(model, 'src/diabetes_model.pkl')
print('Model trained and saved as src/diabetes_model.pkl') 