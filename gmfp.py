import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
food_df = pd.read_csv("wfp_market_food_prices.csv", encoding='latin-1')

print(food_df.head(10))
print("\nUnique countries:\n", food_df["adm0_name"].unique())
print("\nDataset statistics:\n", food_df.describe())

# Missing value analysis
missing = food_df.isnull().sum()
percentage = round(missing / len(food_df) * 100)
print("\nMissing percentage:\n", percentage)

# Drop rows with missing values (~2% in adm1_name)
food_df = food_df.dropna()

# Year vs Price scatter plot
x = food_df['mp_year']
y = food_df['mp_price']
plt.scatter(x, y)
plt.xlabel('Year')
plt.ylabel('Price')
plt.title('Year vs Price')
plt.show()

# Meat subset analysis
meat_subset = food_df[food_df['cm_name'].str.contains('meat', case=False)]

# Meat entries by country
country_counts = meat_subset['adm0_name'].value_counts()
cmap = get_cmap('tab20')
country_counts.plot(kind='bar', figsize=(10, 6), color=cmap(range(len(country_counts))))
plt.title('Number of Entries for Each Country with "Meat"')
plt.xlabel('Country')
plt.ylabel('Number of Entries')
plt.show()

# Meat entries over the years
country_year_counts = meat_subset.groupby('mp_year').size()
country_year_counts.plot(kind='line', marker='o', figsize=(12, 8))
plt.title('Number of Entries Over the Years for Meat')
plt.xlabel('Year')
plt.ylabel('Number of Entries')
plt.grid(True)
plt.show()

# Sale type distribution
print("\nSale types:", food_df["pt_name"].unique())
print("\nSale type counts:\n", food_df["pt_name"].value_counts())

# Encode categorical columns
le = LabelEncoder()
categorical_cols = ['adm0_name', 'adm1_name', 'mkt_name', 'cm_name',
                    'cur_name', 'pt_name', 'um_name', 'mp_commoditysource']
for col in categorical_cols:
    food_df[col] = le.fit_transform(food_df[col])

# Random Forest Regression
X = food_df.drop('mp_price', axis=1)
y = food_df['mp_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nRandom Forest R² (all features): {r2:.4f}")

# Actual vs Predicted prices
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual Prices vs Predicted Prices')
plt.show()
