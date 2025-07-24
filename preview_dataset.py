import pandas as pd

# Read the CSV file
df = pd.read_csv('Multiclass_Diabetes_Dataset.csv')

# Preview the first 5 rows
print('First 5 rows:')
print(df.head())

# Show basic info
df.info()

# Show summary statistics
print('\nSummary statistics:')
print(df.describe()) 