import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# Step 1: Load data from pickle file
pickle_file = 'data.pickle'
with open(pickle_file, 'rb') as f:
    data_dict = pickle.load(f)

data = np.array(data_dict['data'])  # Ensure data is a NumPy array
labels = data_dict['labels']

if not data.size or not labels:
    raise ValueError("The dataset is empty or invalid. Ensure the 'data.pickle' file contains valid data.")

print(f"Loaded {len(data)} samples with {len(labels)} labels.")

# Step 2: Normalize pixel values if data represents raw images
data = data / 255.0  # Normalize pixel values to [0, 1]

# Step 3: Preprocess labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Step 4: Split dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, stratify=labels_encoded, random_state=42)

# Step 5: Define models to evaluate
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    
}

# Step 6: Train and evaluate each model
results = {}
for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train.reshape(len(X_train), -1), y_train)  # Flatten image data if necessary
    y_pred = model.predict(X_test.reshape(len(X_test), -1))
    accuracy = accuracy_score(y_test, y_pred)
    results[model_name] = accuracy
    print(f"{model_name} Accuracy: {accuracy:.2f}")

# Step 7: Display and compare results
print("\nModel Comparison:")
for model_name, accuracy in results.items():
    print(f"{model_name}: {accuracy:.2f}")

# Step 8: Save the best model and LabelEncoder using pickle
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

model_file = f'{best_model_name.replace(" ", "_").lower()}_model.pkl'
with open(model_file, 'wb') as f:
    pickle.dump(best_model, f)
print(f"\nBest model ({best_model_name}) saved to '{model_file}'.")

label_encoder_file = 'label_encoder.pkl'
with open(label_encoder_file, 'wb') as f:
    pickle.dump(label_encoder, f)
print(f"Label encoder saved to '{label_encoder_file}'.")