# 3.1.1) AverageCharge = TotalCharges / tenure   (for all tenure>0)
train_df['AvgCharge'] = train_df.apply(
    lambda row: row['TotalCharges'] / row['tenure'] if row['tenure'] > 0 else row['MonthlyCharges'],
    axis=1
)

# 3.1.2) ChargeDifference = MonthlyCharges - (TotalCharges / tenure)
train_df['ChargeDiff'] = train_df['MonthlyCharges'] - train_df['AvgCharge']



# Mapping categorical values to numeric
train_df['gender'] = train_df['gender'].map({'Male': 1, 'Female': -1})
train_df['SeniorCitizen'] = train_df['SeniorCitizen'].map({0: -1, 1: 1})
train_df['Partner'] = train_df['Partner'].map({'No': -1, 'Yes': 1})
train_df['Dependents'] = train_df['Dependents'].map({'No': -1, 'Yes': 1})
train_df['PaperlessBilling'] = train_df['PaperlessBilling'].map({'No': 1, 'Yes': -1})
# train_df['PhoneService'] = train_df['PhoneService'].map({'No': -1, 'Yes': 1})

# Creating the total_count column by summing the transformed values
train_df['total_count'] = train_df[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PaperlessBilling']].sum(axis=1)

train_df['MultipleLines'] = train_df['MultipleLines'].map({'No': -1, 'Yes': 1, 'No phone service': 0})
train_df['OnlineBackup'] = train_df['OnlineBackup'].map({'No': -1, 'Yes': 1, 'No internet service': 0})
train_df['DeviceProtection'] = train_df['DeviceProtection'].map({'No': -1, 'Yes': 1, 'No internet service': 0})
train_df['StreamingTV'] = train_df['StreamingTV'].map({'No': -1, 'Yes': 1, 'No internet service': 0})
train_df['StreamingMovies'] = train_df['StreamingMovies'].map({'No': -1, 'Yes': 1, 'No internet service': 0})

# Define the same mapping dictionary
gender_map = {'Male': 1, 'Female': -1}
senior_map = {0: -1, 1: 1}
binary_map = {'No': -1, 'Yes': 1}
billing_map = {'No': 1, 'Yes': -1}
multilines_map = {'No': -1, 'Yes': 1, 'No phone service': 0}
internet_map = {'No': -1, 'Yes': 1, 'No internet service': 0}

# Apply to val_df
val_df['gender'] = val_df['gender'].map(gender_map)
val_df['SeniorCitizen'] = val_df['SeniorCitizen'].map(senior_map)
val_df['Partner'] = val_df['Partner'].map(binary_map)
val_df['Dependents'] = val_df['Dependents'].map(binary_map)
val_df['PaperlessBilling'] = val_df['PaperlessBilling'].map(billing_map)
val_df['MultipleLines'] = val_df['MultipleLines'].map(multilines_map)
val_df['OnlineBackup'] = val_df['OnlineBackup'].map(internet_map)
val_df['DeviceProtection'] = val_df['DeviceProtection'].map(internet_map)
val_df['StreamingTV'] = val_df['StreamingTV'].map(internet_map)
val_df['StreamingMovies'] = val_df['StreamingMovies'].map(internet_map)

val_df['total_count'] = val_df[[
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'PaperlessBilling', 'MultipleLines',
    'OnlineBackup', 'DeviceProtection',
    'StreamingTV', 'StreamingMovies'
]].sum(axis=1)

# Apply to test_df
test_df['gender'] = test_df['gender'].map(gender_map)
test_df['SeniorCitizen'] = test_df['SeniorCitizen'].map(senior_map)
test_df['Partner'] = test_df['Partner'].map(binary_map)
test_df['Dependents'] = test_df['Dependents'].map(binary_map)
test_df['PaperlessBilling'] = test_df['PaperlessBilling'].map(billing_map)
test_df['MultipleLines'] = test_df['MultipleLines'].map(multilines_map)
test_df['OnlineBackup'] = test_df['OnlineBackup'].map(internet_map)
test_df['DeviceProtection'] = test_df['DeviceProtection'].map(internet_map)
test_df['StreamingTV'] = test_df['StreamingTV'].map(internet_map)
test_df['StreamingMovies'] = test_df['StreamingMovies'].map(internet_map)

test_df['total_count'] = test_df[[
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'PaperlessBilling', 'MultipleLines',
    'OnlineBackup', 'DeviceProtection',
    'StreamingTV', 'StreamingMovies'
]].sum(axis=1)



val_df['AvgCharge'] = val_df.apply(
    lambda row: row['TotalCharges'] / row['tenure'] if row['tenure'] > 0 else row['MonthlyCharges'],
    axis=1
)

test_df['AvgCharge'] = test_df.apply(
    lambda row: row['TotalCharges'] / row['tenure'] if row['tenure'] > 0 else row['MonthlyCharges'],
    axis=1
)
# val_df['AvgCharge'] = val_df['TotalCharges'] / val_df['tenure']
# test_df['AvgCharge'] = test_df['TotalCharges'] / test_df['tenure']

val_df['ChargeDiff'] = val_df['MonthlyCharges'] - val_df['AvgCharge']
test_df['ChargeDiff'] = test_df['MonthlyCharges'] - test_df['AvgCharge']