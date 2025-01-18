import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Step 1: Load data from pickle file
pickle_file = 'data.pickle'
with open(pickle_file, 'rb') as f:
    data_dict = pickle.load(f)

data = data_dict['data']
labels = data_dict['labels']

# Check if data is available
if not data or not labels:
    raise ValueError("The dataset is empty. Ensure the 'data.pickle' file contains valid data.")

print(f"Loaded {len(data)} samples with {len(labels)} labels.")

# Step 2: Preprocess labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)  # Encode labels as integers

# Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)

# Step 4: Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the trained model for future use
import joblib
model_file = 'random_forest_model.pkl'
joblib.dump(rf_model, model_file)
print(f"Trained model saved to '{model_file}'.")