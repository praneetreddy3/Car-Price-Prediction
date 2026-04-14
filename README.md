# Car-Price-Prediction

**Statistical Modeling of Automobile Prices using Linear Regression and LASSO in R**

## Overview

This project builds **predictive models for car pricing** using multiple linear regression and LASSO (Least Absolute Shrinkage and Selection Operator) regularization in R. The analysis identifies the key factors that drive automobile prices — including engine size, curb weight, car dimensions, and brand — and builds optimized pricing models with feature selection and multicollinearity handling.

**Language:** R
**Models:** Multiple Linear Regression, Stepwise AIC, LASSO (glmnet)
**Dataset:** 205 automobiles with 26 features
**Key Predictors:** Engine size, curb weight, car dimensions, brand

## Dataset

**Source:** CarPrice_Assignment.csv (205 observations, 26 variables)

**Target Variable:** `price` (continuous, USD)

**Key Features:**
- **Numerical:** engine size, curb weight, horsepower, highway MPG, wheelbase, car length/width/height, bore ratio, stroke, compression ratio, peak RPM
- **Categorical:** fuel type (gas/diesel), aspiration (std/turbo), car body (sedan/hatchback/wagon/hardtop/convertible), drive wheel (fwd/rwd/4wd), engine type, cylinder count, fuel system, brand

**Data Dictionary:** Included in `data/Dictionary-carprices.xlsx`

## Methodology

### 1. Exploratory Data Analysis

- Distribution analysis of car prices (right-skewed)
- Correlation heatmap of all numeric variables
- Boxplots for price by fuel type and outlier detection
- Scatterplots revealing strong linear relationships (engine size vs. price)

### 2. Model Building Pipeline

```
Raw Data (205 cars, 26 features)
    |
[Data Preprocessing]
- Convert categorical variables to factors
- Extract brand name from CarName
- Handle typos (toyouta → toyota, vokswagen → volkswagen)
    |
[Model 1: Full Linear Regression]
- All predictors included
- Baseline R² assessment
    |
[Model 2: Stepwise AIC Selection]
- Bidirectional stepwise regression using AIC criterion
- Automated feature selection
- Multicollinearity check via VIF
    |
[Model 3: Refined Model]
- Remove high-leverage observations
- Refit with selected predictors
- Residual diagnostics (Q-Q plot, residuals vs. fitted)
    |
[Model 4: LASSO Regression]
- Cross-validated lambda selection (cv.glmnet)
- L1 regularization for automatic feature selection
- RMSE evaluation on held-out test set
```

### 3. Model Evaluation

**Train/Test Split:** 165 training / 40 test observations

**LASSO Results:**
- Cross-validated optimal lambda selected via `cv.glmnet`
- Automatic shrinkage of irrelevant coefficients to zero
- RMSE reported on test set for generalization assessment

## Key Findings

**Most Influential Price Predictors:**
- Engine size (strongest positive correlation)
- Curb weight (heavier cars cost more)
- Car width and length (larger dimensions = higher price)
- Brand (luxury brands command premium pricing)
- Highway MPG (negative correlation — efficient cars tend to be cheaper)

**Model Improvements:**
- Stepwise AIC reduced model complexity while maintaining predictive power
- High-leverage point removal improved residual normality
- LASSO regularization provided the most parsimonious model with competitive accuracy

## Prerequisites

- R 4.0+
- RStudio (recommended)

### Required R Packages

```r
install.packages(c("readr", "readxl", "dplyr", "ggplot2",
                   "caret", "tidyr", "MASS", "car",
                   "corrplot", "glmnet"))
```

## Usage

### Option 1: RStudio

```r
# Open in RStudio
# File → Open File → src/STAT515-Assignment6.Rmd
# Click "Knit" to run all chunks and generate report
```

### Option 2: Command Line

```bash
Rscript -e "rmarkdown::render('src/STAT515-Assignment6.Rmd')"
```

## Repository Structure

```
Car-Price-Prediction/
├── README.md
├── .gitignore
├── src/
│   └── STAT515-Assignment6.Rmd      # Full R analysis (EDA + modeling)
├── data/
│   ├── CarPrice_Assignment.csv       # Dataset (205 cars, 26 features)
│   └── Dictionary-carprices.xlsx     # Variable definitions
└── docs/
    └── STAT515_Group20_Report.pdf    # Complete analysis report
```

## Visualizations

The analysis produces several key visualizations:

- **Price Distribution Histogram** — Right-skewed distribution of car prices
- **Correlation Heatmap** — Relationships between all numeric features
- **Engine Size vs. Price Scatterplot** — Strongest linear predictor
- **Price by Fuel Type Boxplot** — Diesel vs. gas price comparison
- **Residual Diagnostic Plots** — Model assumption validation (4-panel)

## Limitations & Future Work

**Current Limitations:**
- Small dataset (205 observations)
- Brand name typos required manual correction
- Linear models may underfit nonlinear price relationships

**Future Improvements:**
- Ensemble methods (Random Forest, Gradient Boosting) for nonlinear patterns
- Web scraping for larger, more current pricing data
- Cross-validation for more robust error estimates
- Feature engineering (price-per-horsepower, luxury index)

## Author

**Praneet Chinthala**
- M.S. Data Analytics Engineering, George Mason University
- Email: praneetreddy66@gmail.com
- GitHub: [@praneetreddy3](https://github.com/praneetreddy3)

## License

This project is for educational purposes.

---

**Last Updated:** April 2026
