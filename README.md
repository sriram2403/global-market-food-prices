# 🌍 Global Market Food Prices Analysis

A machine learning project that explores and predicts global food market prices using the World Food Programme (WFP) dataset.

---

## 📋 Overview

This project provides a thorough exploration of global food prices using a real-world dataset. It applies statistical and machine learning approaches to uncover patterns and relationships, then evaluates and compares multiple regression models for price prediction.

**Key Stakeholder:** The FDA (Food and Drug Administration), which monitors meat prices to assess economic factors impacting food safety and regulatory decisions.

---

## 🎯 Objectives

- Explore global food price trends through data analysis and visualization
- Uncover patterns and relationships within the WFP dataset
- Evaluate and compare machine learning regression models
- Recommend methods for improving model performance

---

## 📁 Repository Structure

```
├── gmfp.ipynb                     # Main Jupyter Notebook
├── README.md
└── wfp_market_food_prices.csv     # Dataset (download separately — see below)
```

---

## 📊 Dataset

**Source:** [WFP Global Food Prices — Kaggle](https://www.kaggle.com/datasets/jboysen/global-food-prices)

Download the dataset from Kaggle and place `wfp_market_food_prices.csv` in the root directory before running the notebook.

**Key columns:**
| Column | Description |
|---|---|
| `adm0_name` | Country name |
| `adm1_name` | State/region name |
| `mkt_name` | Market name |
| `cm_name` | Commodity (food item) |
| `pt_name` | Sale type (Retail, Producer, etc.) |
| `mp_year` | Year of price data |
| `mp_price` | Market price (target variable) |

---

## 🔍 Analysis Highlights

### Data Cleaning
- Only `adm1_name` (state name) had missing values (~2% of data)
- Missing rows were dropped with minimal impact on dataset integrity

### Visualizations
- **Year vs. Price:** Meat prices peaked in 2000–2003, dipped, then rose again around 2012
- **Meat by Country:** Lao PDR leads in meat market entries; Timor-Leste has the fewest
- **Meat Sales Over Time:** Sales climbed steadily to a peak in 2015, then declined
- **Sale Type Distribution:** Retail stores dominate sales volume; Producer stores have the fewest

### Models Compared
| Model | Performance |
|---|---|
| **Random Forest Regressor** | ✅ Best — near-ideal fit between actual and predicted prices |
| Decision Tree Regressor | Good for lower prices; struggles with higher-priced items |
| XGBoost Regressor | Competitive but below Random Forest |
| AdaBoost Regressor | Lowest among the four |

---

## 🛠️ Installation & Usage

### Requirements
```bash
pip install pandas numpy matplotlib scikit-learn scipy
```

### Run the Notebook
```bash
jupyter notebook gmfp.ipynb
```

Make sure `wfp_market_food_prices.csv` is in the same directory as the notebook.

---

## 📌 Key Findings

- Meat prices have generally increased over time across most countries
- Countries such as Lao PDR, Rwanda, and Democratic Republic of Congo show high meat consumption
- **Random Forest Regression** outperforms all other models tested, making it the recommended approach for food price prediction

---

## 📚 References

- Dataset: [WFP Global Food Prices on Kaggle](https://www.kaggle.com/datasets/jboysen/global-food-prices)
- World Food Programme (WFP) — [wfp.org](https://www.wfp.org)
