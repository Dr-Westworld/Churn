# Chrun
<img width="1900" height="890" alt="Screenshot 2026-06-26 114611" src="https://github.com/user-attachments/assets/849ce1bf-a65e-4d83-a0e0-09a26dc1cb20" />
<img width="1917" height="1001" alt="Screenshot 2026-06-26 114743" src="https://github.com/user-attachments/assets/d50c6096-2f91-4330-976e-17950748cc87" />
<img width="1918" height="1078" alt="image" src="https://github.com/user-attachments/assets/f062aa41-2f4a-4c2f-8b7f-d9fc97c0917b" />



## Xgboost
![image](https://github.com/user-attachments/assets/becf4ad5-d714-42ff-bf9d-6a8e9d8bab84)

(with eda like this plus handel nan value by calculating tenture and monthly charges to fill the nan value )
```python
# mapping dictionary
gender_map = {'Male': 1, 'Female': -1}
senior_map = {0: -1, 1: 1}
binary_map = {'No': -1, 'Yes': 1}
billing_map = {'No': 1, 'Yes': -1}
multilines_map = {'No': -1, 'Yes': 1, 'No phone service': 0}
internet_map = {'No': -1, 'Yes': 1, 'No internet service': 0}

df['gender'] = df['gender'].map(gender_map)
df['SeniorCitizen'] = df['SeniorCitizen'].map(senior_map)
df['Partner'] = df['Partner'].map(binary_map)
df['Dependents'] = df['Dependents'].map(binary_map)
df['PaperlessBilling'] = df['PaperlessBilling'].map(billing_map)
df['MultipleLines'] = df['MultipleLines'].map(multilines_map)
df['OnlineBackup'] = df['OnlineBackup'].map(internet_map)
df['DeviceProtection'] = df['DeviceProtection'].map(internet_map)
df['StreamingTV'] = df['StreamingTV'].map(internet_map)
df['StreamingMovies'] = df['StreamingMovies'].map(internet_map)

df['total_count'] = df[[
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'PaperlessBilling', 'MultipleLines',
    'OnlineBackup', 'DeviceProtection',
    'StreamingTV', 'StreamingMovies'
]].sum(axis=1)

df['AvgCharge'] = df.apply(
    lambda row: row['TotalCharges'] / row['tenure'] if row['tenure'] > 0 else row['MonthlyCharges'],
    axis=1
)
df['ChargeDiff'] = df['MonthlyCharges'] - df['AvgCharge']
```

## Lightbgm
(only handel nan value by eda no other mathematical use)
``Fitting 5 folds for each of 720 candidates, totalling 3600 fits``
![image](https://github.com/user-attachments/assets/073d1a09-7153-4e93-9747-13a755164591)
