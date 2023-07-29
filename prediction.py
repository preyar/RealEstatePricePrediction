# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample dataset of house prices and areas (replace this with your own dataset)
data = {
    'Area': [1000, 1500, 1200, 1800, 900, 1100, 1300, 1600],
    'Price': [300000, 450000, 360000, 540000, 270000, 330000, 390000, 480000]
}

# Create a DataFrame from the dataset
df = pd.DataFrame(data)

# Split the data into features (X) and target (y)
X = df[['Area']]
y = df['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# Get area as user input for prediction
while True:
    try:
        user_input_area = float(input("Enter the area in sq. ft for price prediction: "))
        break
    except ValueError:
        print("Invalid input! Please enter a valid number.")

# Predict the price for the user input area
predicted_price = model.predict([[user_input_area]])
print(f"Predicted Price for {user_input_area} sq. ft: {predicted_price[0]}")
