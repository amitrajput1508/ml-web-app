import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Generate sample data (Square Footage vs Price)
np.random.seed(42)
square_feet = np.random.randint(500, 4000, 100)
price = square_feet * 150 + np.random.randint(10000, 50000, 100)  # Basic linear relationship

# Convert to DataFrame
df = pd.DataFrame({'SquareFeet': square_feet, 'Price': price})

# Split into training and testing sets
X = df[['SquareFeet']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open('model/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")
