import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(data/Housing_Rent_NZ)

#Cleaning Data
print(df.shape)
print(df.dtypes)
print(df.isna().sum())
print((df.isna().sum() / len(df) * 100).round(2))
# NA - rent_price 2% and property_type 1.49%

print(df["Region"].value_counts(dropna=False).head(50))
print(df["Property_Type"].value_counts(dropna=False).head(50))

print(sorted(df["Region"].dropna().unique())[:100])
print(sorted(df["Property_Type"].dropna().unique())[:100])

df["Region_cl"] = (
    df["Region"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
)

df["Region_cl"] = df["Region_cl"].str.title()

df["Property_Type_cl"] = (
    df["Property_Type"]
    .astype(str)
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
    .str.title()
)

corrections = {
    "Aprtment": "Apartment",
    "aprtment": "Apartment",
    "apartment": "Apartment",
    "Aptartment": "Apartment",
    "HOUSE": "House",
    "house": "House",
    "Town House": "Townhouse",
    "townhouse": "Townhouse",
    "Nan": np.nan,
}
df["Property_Type_cl"] = df["Property_Type_cl"].replace(corrections)


region_corrections = {
    "Welling Ton": "Wellington",
    "WellingTon": "Wellington",
    "Welling ton": "Wellington",
    "AUCKLAND": "Auckland",
    "auckland": "Auckland",
    "auckland ": "Auckland",
    "CHRISTCHURCH": "Christchurch",
    "christchurch": "Christchurch",
    "DUNEDIN": "Dunedin",
    "Dunedin  ": "Dunedin",
    "dunedin": "Dunedin",
    "HAMILTON": "Hamilton",
    "hamilton": "Hamilton",
    "Wellington": "Wellington",
    "Auckland": "Auckland",
    "Christchurch": "Christchurch",
    "Hamilton": "Hamilton",
    "Dunedin": "Dunedin",
}

df["Region_cl"] = df["Region_cl"].replace(region_corrections)

print(df["Region_cl"].unique())

df_cl = df.drop(["Region", "Property_Type"], axis=1)
df_cl = df_cl.rename(
    columns={"Region_cl": "Region", "Property_Type_cl": "Property_Type"}
)

df_cl = df_cl.dropna()

df_cl = df_cl[
    ["Date", "Region", "Property_Type"]
    + [col for col in df_cl.columns if col not in ["Date", "Region", "Property_Type"]]
]

# Rent Wisker Plot
df_cl["Rent_Price"].describe()
plt.figure(figsize=(10, 5))
sns.boxplot(x="Region", y="Rent_Price", data=df_cl)
plt.title("Rent price by Region")
plt.tight_layout()
plt.show()

# Outliers
Q1 = df_cl["Rent_Price"].quantile(0.25)
Q3 = df_cl["Rent_Price"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers = df_cl[(df_cl["Rent_Price"] < lower) | (df_cl["Rent_Price"] > upper)]
print("Outliers count:", len(outliers))
outliers.sort_values("Rent_Price", ascending=False).head(20)

print(df_cl.head(20))
miss_data = df_cl.isna().sum()
print(miss_data)

# Line Graphs
df_ts = df_cl.copy()
df_ts["Date"] = pd.to_datetime(df_ts["Date"])
df_ts["Year"] = df_ts["Date"].dt.year
df_ts["Month"] = df_ts["Date"].dt.month
df_ts["Month_Name"] = df_ts["Date"].dt.strftime("%b")
df_ts["Year_Month"] = df_ts["Date"].dt.to_period("M")

monthly_avg = df_ts.groupby("Year_Month")["Rent_Price"].mean().reset_index()
yearly_avg = df_ts.groupby("Year")["Rent_Price"].mean().reset_index()

monthly_avg["Year_Month_Label"] = monthly_avg["Year_Month"].dt.strftime("%Y-%b")

print("Average Rent Price by Year:")
print(yearly_avg)
print("\nAverage Rent Price by Month (first 12 months):")
print(monthly_avg[["Year_Month_Label", "Rent_Price"]].head(12))
plt.figure(figsize=(20, 12))

plt.subplot(2, 3, 1)
plt.plot(
    monthly_avg["Year_Month_Label"],
    monthly_avg["Rent_Price"],
    marker="o",
    linewidth=2,
    markersize=4,
    color="#9ccaeb",
    alpha=0.8,
)
plt.title("Rent Price Trend by Month", fontsize=14, fontweight="bold")
plt.xlabel("Year-Month")
plt.ylabel("Average Rent Price")
plt.xticks(rotation=90)
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 2)
plt.plot(
    yearly_avg["Year"].astype(int),
    yearly_avg["Rent_Price"],
    marker="s",
    linewidth=3,
    markersize=8,
    color="#9274A7",
    alpha=0.8,
)
plt.title("Rent Price Trend by Year", fontsize=14, fontweight="bold")
plt.xlabel("Year")
plt.ylabel("Average Rent Price")
plt.xticks(yearly_avg["Year"].astype(int))
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 3)
colors = ["#9ccaeb", "#6dbd7e", "#c7ccc7", "#eba360", "#9274A7"]
for i, region in enumerate(df_ts["Region"].unique()[:5]):
    region_data = df_ts[df_ts["Region"] == region]
    region_yearly = region_data.groupby("Year")["Rent_Price"].mean()
    plt.plot(
        region_yearly.index.astype(int),
        region_yearly.values,
        marker="o",
        label=region,
        linewidth=2,
        color=colors[i],
        alpha=0.8,
    )
plt.title("Rent Price Trend by Region", fontsize=14, fontweight="bold")
plt.xlabel("Year")
plt.ylabel("Average Rent Price")
plt.legend()
plt.xticks(region_yearly.index.astype(int))
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Descriptive stat
df_cl.groupby("Region")["Rent_Price"].describe()
df_cl.groupby("Property_Type")["Rent_Price"].describe()
df_cl.groupby(["Region", "Property_Type"])["Rent_Price"].median()

print("Overall Rent Price Stats:")
print(df_cl["Rent_Price"].describe())
print("\n")

print("Stats by Region:")
print(df_cl.groupby("Region")["Rent_Price"].describe())
print("\n")

print("Stats by Property Type:")
print(df_cl.groupby("Property_Type")["Rent_Price"].describe())
print("\n")

print("Median Rent Price by Region & Property Type:")
print(df_cl.groupby(["Region", "Property_Type"])["Rent_Price"].median())

# Correlation
corr = df_cl.corr(numeric_only=True)

print("Correlation matrix:")
print(corr)

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

X = pd.get_dummies(
    df_cl[["Bedrooms", "Distance_to_CBD_km", "Property_Type"]], drop_first=True
)
y = df_cl["Rent_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("R2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Rent Price")
plt.ylabel("Predicted Rent Price")
plt.title("Regression with Property Type")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.show()

import statsmodels.api as sm

df_reg = df_cl.copy()

df_dummies = pd.get_dummies(
    df_reg[["Property_Type", "Region"]], drop_first=True, dtype=float
)

X = df_dummies.astype(float)
X = sm.add_constant(X)

y = pd.to_numeric(df_reg["Rent_Price"], errors="coerce").fillna(method="ffill")

min_len = min(len(X), len(y))
X = X.iloc[:min_len]
y = y.iloc[:min_len]

model = sm.OLS(y, X).fit()
print(model.summary())

# Coefficients
coeffs = pd.DataFrame(
    {
        "Variable": model.params.index,
        "Coefficient": model.params.values,
        "P-value": model.pvalues.values,
    }
)
print("\nCoefficients:")
print(coeffs.sort_values(by="Coefficient", ascending=False))

# ARIMA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

df_ts = df_cl.copy()
df_ts['Date'] = pd.to_datetime(df_ts['Date'])
df_ts = df_ts.sort_values('Date')

monthly_data = df_ts.groupby(pd.Grouper(key='Date', freq='M'))['Rent_Price'].mean().dropna()
ts_values = monthly_data.values

train_size = int(len(ts_values) * 0.8)
train, test = ts_values[:train_size], ts_values[train_size:]
train_dates, test_dates = monthly_data.index[:train_size], monthly_data.index[train_size:]

def evaluate_arima_model(train, test, order=(1,1,1)):
     try:
        model = ARIMA(train, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(test))
        mae = mean_absolute_error(test, forecast)
        mape = mean_absolute_percentage_error(test, forecast) * 100
        rmse = np.sqrt(np.mean((test - forecast) ** 2))
        return forecast, model_fit, mae, mape, rmse
     except Exception as e:
        print(f"ERROR ARIMA{order}: {e}")
        return None, None, None, None, None
orders = [(1,1,1), (1,1,0), (2,1,1), (1,0,1)]
best_mae = float('inf')
best_order = None
best_forecast = None
best_model = None

for order in orders:
    forecast, model_fit, mae, mape, rmse = evaluate_arima_model(train, test, order)

if mae is not None and mae < best_mae:
        best_mae = mae
        best_order = order
        best_forecast = forecast
        best_model = model_fit

if mae is not None:
        print(f"ARIMA{order} - MAE: ${mae:.2f}, MAPE: {mape:.1f}%, RMSE: ${rmse:.2f}")

print(f"\nThe best model: ARIMA{best_order}")
print(f"Metrics of the best model:")
print(f"   MAE: ${best_mae:.2f}")
print(f"   MAPE: {mean_absolute_percentage_error(test, best_forecast) * 100:.1f}%")
print(f"   RMSE: ${np.sqrt(np.mean((test - best_forecast) ** 2)):.2f}")

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(monthly_data.index, monthly_data.values, label='Current data', color="#9ccaeb", linewidth=2)
plt.plot(test_dates, best_forecast, label='ARIMA Prediction', color="#016612", linewidth=2, marker='o')
plt.fill_between(test_dates, 
                 best_forecast - best_mae, 
                 best_forecast + best_mae, 
                 alpha=0.2, color="#911414", label='Forecast error')
plt.title(f'ARIMA{best_order} - Rent Price Forecast ', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('AVG Rent Price($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

plt.subplot(2, 2, 2)
future_steps = 6  
future_forecast = best_model.forecast(steps=len(test) + future_steps)
future_dates = pd.date_range(start=test_dates[0], periods=len(test) + future_steps, freq='M')

plt.plot(monthly_data.index, monthly_data.values, label='Historical data', color='#9ccaeb', linewidth=2)
plt.plot(future_dates[:len(test)], best_forecast, label='Forecast (test)', color="#e97f06", linewidth=2)
plt.plot(future_dates[len(test):], future_forecast[len(test):], 
         label='Forecast for the future', color="#0d750d", linewidth=2, linestyle='--')
plt.title('Forecast for the future', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('AVG Rent Price($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


