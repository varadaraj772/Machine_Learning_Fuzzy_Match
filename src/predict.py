import joblib
import pandas as pd

# Load the trained model
clf = joblib.load('fuzzy_model.pkl')

# Input data for prediction
import pandas as pd

# Input data that should yield a score close to 0
input_data = pd.DataFrame({
    'string_1': ['Elephant', 'Rocket', 'Book'],
    'string_2': ['Table', 'Moon', 'Car']
})

# Followed by the prediction steps
predictions = clf.predict(input_data)
print(predictions)  # This should ideally output [0, 0, 0] or similar


# Combine text columns
input_combined = input_data['string_1'] + " " + input_data['string_2']

# Make predictions
predictions = clf.predict(input_combined)

# Output predictions
print(predictions)

probabilities = clf.predict_proba(input_data)
print(probabilities)