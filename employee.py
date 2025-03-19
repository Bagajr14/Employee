import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree

# Sample data (you should replace this with your real employee dataset)
data = {
    'Age': [30, 45, 23, 36, 50, 40, 60, 28],
    'Tenure': [5, 10, 1, 3, 20, 4, 25, 2],
    'SatisfactionLevel': [0.8, 0.6, 0.7, 0.9, 0.5, 0.4, 0.3, 0.8],
    'Salary': ['Low', 'High', 'Low', 'Medium', 'High', 'Low', 'Medium', 'High'],
    'Department': ['Sales', 'HR', 'IT', 'Sales', 'HR', 'IT', 'Sales', 'HR'],
    'Left': [0, 1, 0, 1, 0, 1, 0, 1]  # 0 = Stayed, 1 = Left
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Preprocessing: Convert categorical columns to numeric
df['Salary'] = df['Salary'].map({'Low': 0, 'Medium': 1, 'High': 2})
df['Department'] = df['Department'].map({'Sales': 0, 'HR': 1, 'IT': 2})

# Define features (X) and target (y)
X = df[['Age', 'Tenure', 'SatisfactionLevel', 'Salary', 'Department']]
y = df['Left']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the model
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Visualizing the decision tree
plt.figure(figsize=(12, 8))
tree.plot_tree(model, filled=True, feature_names=X.columns, class_names=['Stayed', 'Left'], rounded=True)
plt.show()
