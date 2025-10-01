# Housing_Rent_NZ

## 📊 Project Overview
Comprehensive data analysis of New Zealand rental market from 2022-2024, examining price drivers, market stability, and predictive forecasting.

##📁 Repository structure
- `Housing_Rent_NZ.csv` — original synthetic dataset  
- `Housing_Rent_NZ.py` — Full Python Code
- `Plot.png` - Wisker Plot
- `Line Graphs.png` — Line Graphs with trends
- `Correlation.png` - Correlation 
- `Regression.png` - Multiple Linear Regression
- `ARIMA.png` - ARIMA Forecasting Visualisation
- `Report Housing Rent NZ.pdf` — short business report (insights + recommendations)

## 🎯 Key Findings

### 📈 Market Stability & Predictability
- **Exceptional Market Stability**: ARIMA model achieves 99.4% forecasting accuracy (MAPE: 0.6%)
- **Highly Predictable**: Prices show consistent patterns with minimal changeability
- **Long-term Planning**: Reliable 6 month forecast

### 🏠 Price Drivers Analysis
- **Property Type**: 
  - Houses: +$94.71 vs apartments
  - Townhouses: +$50.14 vs apartments
- **Location Impact**: Distance to CBD has minimal effect (-0.05 correlation)
- **Number of Bedrooms**: has a positive correlation with rent prices (0.44)

 ### 📍 Geographic Insights
- Regional variations are statistically insignificant in the current model
- The market behaves consistently across different locations
- Focus on property characteristics over location for pricing

## 🛠 Methodology
 **Data Cleaning**: Handled missing values, duplicates, outliers, and data type conversions
- **Feature Engineering**: Created temporal features and categorical encodings
- **Quality Control**: Ensured data integrity throughout pipeline

### 📉 Analytical Techniques
1. **Descriptive Statistics**: Market overview and distribution analysis
2. **Correlation Analysis**: Identified relationship between variables
3. **Multiple Regression**: Quantified impact of price drivers
4. **Time Series Analysis (ARIMA)**: Market stability and forecasting

### 🔍 Models Used
- **Linear Regression**: R² = 0.325 (32.5% variance explained)
- **ARIMA(1,0,1)**: MAE = $3.64, MAPE = 0.6%, RMSE = $4.54
- **Statistical Tests**: p-value validation of coefficients

