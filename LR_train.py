import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed data
data = pd.read_csv(r"C:\Users\PHQ\Machine-Learning-Project\data\final_mental_health_data.csv")

# Prepare X (features) and y (target)
X = data.drop('Severity', axis=1)
y = data['Severity']

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Verify the shape of your data split
print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# Initialize Logistic Regression model with class weights to handle class imbalance
model = LogisticRegression(class_weight='balanced',  max_iter=5000, solver='lbfgs')

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model performance
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Class 0','Class 1','Class 2'],
            yticklabels=['Class 0','Class 1','Class 2'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

# Save the figure for your PPT or report
plt.savefig('confusion_matrix_lr.png', dpi=300, bbox_inches='tight')

