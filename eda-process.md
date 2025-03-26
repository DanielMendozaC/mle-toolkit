# Exploratory Data Analysis (EDA) Process

A systematic approach to understanding your data before modeling.

## 1. Initial Data Overview
- Check dataset shape: `df.shape`
- View sample data: `df.head()`, `df.tail()` 
- Examine data types: `df.info()`
- Get basic statistics: `df.describe()`, `df_describe(include='all').T`

## 2. Data Quality Assessment
- Find missing values: `df.isna().sum()`
- Check duplicates: `df.duplicated().sum()`
- Identify outliers using boxplots or z-scores

## 3. Distribution Analysis
- Plot numerical distributions: histograms, box plots
- Examine categorical distributions: value counts, bar charts
- Check for skewness in numerical features

## 4. Relationship Exploration
- Create correlation matrix: `df.corr()`
- Visualize correlations with heatmap
- Explore feature relationships with scatter plots

## 5. Target Variable Analysis
- Examine target distribution
- Check relationship between features and target
- Identify most predictive features

## 6. Preprocessing Insights
- Document required transformations (scaling, encoding)
- Note features needing special handling
- Plan feature engineering approach
