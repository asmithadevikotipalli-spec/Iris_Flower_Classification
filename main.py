import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load and prepare the data
iris = load_iris()
X = iris.data
y = iris.target

# Create a DataFrame for visualization purposes
df = pd.DataFrame(data=X, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in y]

# 2. Visualize the data
# This creates the Pairplot using the DataFrame
sns.pairplot(df, hue='species', palette='viridis')
plt.show()

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 5. Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=iris.target_names))