import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import numpy as np

# Named function for splitting strings instead of lambda
def analyzer_func(text):
    return text.split()

# Load dataset
df = pd.read_csv('data/dataset.csv')

# Handle missing data (if any)
df.dropna(subset=['string_1', 'string_2', 'match'], inplace=True)

# Train-test split
X = df[['string_1', 'string_2']]
y = df['match']

# Use TF-IDF Vectorizer with the named function
vectorizer = TfidfVectorizer(analyzer=analyzer_func)
X_tfidf = vectorizer.fit_transform(X['string_1'] + ' ' + X['string_2'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Define the RandomForestClassifier with class_weight
model = RandomForestClassifier(class_weight='balanced', random_state=42)

# Optional: Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)
model = grid_search.best_estimator_

# Train the model
model.fit(X_train, y_train)

# Save the model and vectorizer
with open('models/fuzzy_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
