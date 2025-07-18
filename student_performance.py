import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("student-data.csv")

# Features and target
X = df[['StudyHours', 'Attendance', 'PreviousScore']]
y = df['FinalGrade']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Display results
for i in range(len(X_test)):
    print(f"Predicted: {predictions[i]:.2f}, Actual: {y_test.values[i]}")

# Visualize
plt.scatter(X['StudyHours'], y, color='blue', label='Actual')
plt.scatter(X['StudyHours'], model.predict(X), color='red', label='Predicted')
plt.xlabel('Study Hours')
plt.ylabel('Final Grade')
plt.title('Study Hours vs Final Grade')
plt.legend()
plt.grid()
plt.show()
