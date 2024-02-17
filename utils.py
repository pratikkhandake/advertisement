# utils.py

# Import necessary libraries
import pickle

# File path of the trained ridge regression model
MODEL_FILE_PATH = "Advertisement.pkl"

# Function to load the trained ridge regression model
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Function to get predicted sales using the loaded model
def get_predicted_sales(tv, radio, newspaper):
    # Load the trained ridge regression model
    model = load_model(MODEL_FILE_PATH)

    # Make prediction using the model
    predicted_sales = model.predict([[tv, radio, newspaper]])

    # Return the predicted sales
    return predicted_sales[0]
