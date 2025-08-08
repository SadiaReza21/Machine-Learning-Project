import pandas as pd
from sklearn.preprocessing import StandardScaler

# Step 1: Load the initial dataset
df = pd.read_csv(r"D:\mental_health_dataset\mental_health_data.csv")

# Step 2: Handle missing values (Remove rows where 'Severity' is "None")
# Replace "None" with NaN in the 'Severity' column
df['Severity'].replace('None', pd.NA, inplace=True)

# Drop rows with missing 'Severity' values
df.dropna(subset=['Severity'], inplace=True)

# Save the cleaned data to a new file
df.to_csv(r"D:\mental_health_dataset\cleaned_mental_health_data.csv", index=False)

# Step 3: Apply scaling (Standardization) to numerical features
scaler = StandardScaler()

# List of numerical columns to scale
numerical_cols = ['Age', 'Sleep_Hours', 'Work_Hours', 'Physical_Activity_Hours', 'Social_Media_Usage']

# Apply scaling (Standardization)
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Save the final scaled dataset to a new file
df.to_csv(r"D:\mental_health_dataset\final_mental_health_data.csv", index=False)

print("Preprocessing complete: Data cleaned and scaled, saved to final file.")


# Descriptive statistics for the scaled numerical features
print("\nDescriptive Statistics for Scaled Data:")
print(df[['Age', 'Sleep_Hours', 'Work_Hours', 'Physical_Activity_Hours', 'Social_Media_Usage']].describe())

# Check min and max values for scaled columns
print("\nMin Values After Scaling:")
print(df[['Age', 'Sleep_Hours', 'Work_Hours', 'Physical_Activity_Hours', 'Social_Media_Usage']].min())

print("\nMax Values After Scaling:")
print(df[['Age', 'Sleep_Hours', 'Work_Hours', 'Physical_Activity_Hours', 'Social_Media_Usage']].max())