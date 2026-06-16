# ✈️ Flight Delay Analysis and Prediction Using Machine Learning

An end-to-end machine learning and analytics project focused on analyzing historical airline delays and building predictive models to forecast average arrival delays across U.S. airports and airlines.

---

# 📌 Project Overview

Flight delays cost the aviation industry billions of dollars annually through operational disruptions, increased fuel consumption, staffing inefficiencies, and reduced passenger satisfaction.

This project leverages historical flight performance data from the U.S. Bureau of Transportation Statistics (BTS) to:

- Analyze historical delay patterns
- Identify key causes of flight delays
- Engineer operational and delay-related features
- Build predictive machine learning models
- Compare model performance across multiple algorithms
- Generate actionable business insights for airline operations

---

# 🎯 Business Problem

Airlines and airports need accurate delay forecasts to improve operational efficiency and customer experience.

Unexpected delays impact:

- Gate scheduling
- Crew allocation
- Aircraft utilization
- Passenger connections
- Airport congestion

The goal of this project is to predict average arrival delay minutes and provide decision-makers with insights that support proactive planning and resource optimization.

---

# 📊 Dataset

### Source

U.S. Bureau of Transportation Statistics (BTS)
Link: https://www.transtats.bts.gov/OT_Delay/ot_delaycause1.asp?qv52ynB=pun46&20=E

### Time Period

June 2003 – July 2025

### Dataset Size

- 409,612 records
- 21 attributes

### Key Features

#### Operational Variables

- Carrier
- Airport
- Arrival Flights
- Arrival Delays
- Delay Counts

#### Delay Attribution Variables

- Carrier Delay
- Weather Delay
- NAS Delay
- Security Delay
- Late Aircraft Delay

#### Target Variable

Average Arrival Delay (Minutes)

---

# 🧹 Data Cleaning & Preprocessing

Performed extensive preprocessing to improve data quality and model performance.

### Data Cleaning

✅ Removed missing values

✅ Imputed numerical delay fields with zero

✅ Replaced missing categorical values using mode

✅ Removed invalid records

✅ Eliminated negative delay values

✅ Filtered records with zero flight volume

---

# ⚙️ Feature Engineering

Created multiple business-focused features:

### Target Transformation

Converted:

```text
Total Arrival Delay Minutes
```

into:

```text
Average Arrival Delay Per Flight
```

### Engineered Features

📈 Total Delay Causes

```text
Carrier Delay
+ Weather Delay
+ NAS Delay
+ Security Delay
+ Late Aircraft Delay
```

📊 Delay Rate

```text
Delay Rate = Delayed Flights / Total Flights
```

✈️ Flights Per Day

```text
Flights Per Day = Arrival Flights / 30
```

These features significantly improved predictive performance.

---

# 📈 Exploratory Data Analysis

The following business questions were investigated:

### Airline Performance

- Which airlines experience the highest delays?
- Which airlines contribute most to overall delay minutes?

### Airport Operations

- Which airports experience the highest delays?
- Which airports process the largest flight volumes?

### Delay Drivers

- What are the primary causes of delays?
- How do weather and carrier-related delays compare?

### Seasonal Trends

- How do delays vary by month?
- How do delays evolve over time?

### Operational Relationships

- Relationship between flight volume and delays
- Correlation analysis between operational metrics

---

# 📊 Key Visualizations

### Delay Trends Over Time

Analyzed long-term changes in flight delay behavior.

### Airline Delay Analysis

Compared delay performance across carriers.

### Delay Cause Breakdown

Identified major contributors to arrival delays.

### Monthly Seasonality

Evaluated seasonal delay patterns.

### Airport Delay Analysis

Ranked airports by average delay performance.

### Correlation Heatmap

Identified relationships between operational variables.

### Delay Distribution Analysis

Evaluated skewness and outlier behavior.

---

# 🤖 Machine Learning Models

Multiple regression models were trained and evaluated.

### Linear Regression

Baseline model.

Limitations:

- Assumes linear relationships
- Unable to capture complex interactions

### Random Forest Regressor

Advantages:

- Handles non-linear relationships
- Strong baseline performance

Limitations:

- Computationally expensive
- Less efficient for high-dimensional features

### XGBoost Regressor

Advantages:

- Captures complex interactions
- Strong predictive performance

### LightGBM Regressor

Advantages:

- Faster training
- Efficient with large datasets

### CatBoost Regressor

Advantages:

- Native categorical variable handling
- Minimal preprocessing
- Strongest overall performance

---

# 📊 Model Performance Comparison

| Model | RMSE | MAE | R² |
|---------|---------|---------|---------|
| Linear Regression | 1517.37 | 752.98 | 0.9691 |
| Random Forest | 1105.19 | 474.29 | 0.9836 |
| XGBoost | 872.68 | 396.13 | 0.9898 |
| LightGBM | 880.38 | 408.24 | 0.9896 |
| CatBoost | 849.28 | 390.96 | 0.9903 |

🏆 Best Model: CatBoost Regressor

---

# 🔍 Model Evaluation

Performance was assessed using:

### R² Score

Measures explained variance.

Best Result:

```text
R² = 0.9903
```

### RMSE

Measures average prediction error magnitude.

Best Result:

```text
RMSE = 849.28
```

### MAE

Measures average absolute prediction error.

Best Result:

```text
MAE = 390.96
```

### Actual vs Predicted Analysis

Compared model predictions against actual delay values.

### Error Distribution Analysis

Evaluated residual behavior and model stability.

---

# 💡 Key Insights

### 📈 Airline Delays

Certain carriers consistently contribute more to delay minutes than others.

### 🌦 Weather Impact

Weather remains one of the strongest external drivers of delay variability.

### 🏢 Airport Bottlenecks

High-volume airports experience greater delay risk due to congestion effects.

### 📅 Seasonal Effects

Delay patterns exhibit strong monthly seasonality.

### 📊 Operational Efficiency

Delay rates increase significantly as operational load rises.

---

# 🚀 Business Impact

This solution can support:

### Airlines

- Crew scheduling optimization
- Aircraft allocation planning
- Route performance monitoring

### Airports

- Gate management
- Resource allocation
- Congestion forecasting

### Operations Teams

- Early delay risk detection
- Proactive disruption management

### Passengers

- Improved delay communication
- Better travel planning

---

# 🛠 Technologies Used

### Programming

- Python

### Data Analysis

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-Learn
- XGBoost
- LightGBM
- CatBoost

### Statistical Analysis

- Correlation Analysis
- Regression Modeling
- Error Analysis

---

# 📂 Repository Structure

```text
Flight-Delay-Analysis-and-Prediction
│
├── notebooks
│   ├── flight_delay_analysis_prediction.ipynb
│   └── flight_delay_model_experiments.ipynb
│
├── presentation
│   └── flight_delay_analysis_presentation.pptx
│
├── screenshots
│   ├── delay_trends.png
│   ├── airline_delays.png
│   ├── delay_breakdown.png
│   ├── correlation_heatmap.png
│   ├── model_comparison.png
│   ├── actual_vs_predicted.png
│   └── error_distribution.png
│
├── requirements.txt
└── README.md
```

---

# 🌟 Project Highlights

✔ Large-scale aviation dataset (400K+ records)

✔ End-to-end machine learning pipeline

✔ Feature engineering for operational analytics

✔ Advanced ensemble learning models

✔ Model comparison and evaluation

✔ Real-world business application

✔ Data storytelling and visualization

---

# 👤 Author

**Bhavana Bajjuri**

Data Analytics | Machine Learning | Business Intelligence | Data Visualization

Feel free to connect and provide feedback on the project.
