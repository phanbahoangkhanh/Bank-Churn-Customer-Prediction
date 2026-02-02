# 📁 Dataset Folder

## About the Dataset

This folder contains the **Bank Churners** dataset used for credit card customer churn prediction.

---

## 📊 Dataset Information

**File**: `BankChurners.csv`

**Source**: Kaggle - [Credit Card Customers Dataset](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)

**Statistics**:
- **Total Records**: 10,127 customers
- **Features**: 23 columns
- **Target Variable**: `Attrition_Flag`
  - Existing Customer: 8,500 (83.93%)
  - Attrited Customer: 1,627 (16.07%)
- **File Size**: ~1.8 MB

---

## 📋 Column Description

### Target Variable
- **Attrition_Flag**: Customer status (Existing Customer / Attrited Customer)

### Demographic Features
- **Customer_Age**: Age of the customer
- **Gender**: M (Male) / F (Female)
- **Dependent_count**: Number of dependents
- **Education_Level**: Education category
- **Marital_Status**: Marital status
- **Income_Category**: Annual income range

### Account Information
- **Card_Category**: Type of card (Blue, Silver, Gold, Platinum)
- **Months_on_book**: Period of relationship with bank
- **Total_Relationship_Count**: Total number of products held
- **Months_Inactive_12_mon**: Number of months inactive in last 12 months
- **Contacts_Count_12_mon**: Number of contacts in last 12 months

### Financial Metrics
- **Credit_Limit**: Credit limit on the card
- **Total_Revolving_Bal**: Total revolving balance
- **Avg_Open_To_Buy**: Open to buy credit line (Average)
- **Total_Amt_Chng_Q4_Q1**: Change in transaction amount (Q4 vs Q1)
- **Total_Trans_Amt**: Total transaction amount (Last 12 months)
- **Total_Trans_Ct**: Total transaction count (Last 12 months)
- **Total_Ct_Chng_Q4_Q1**: Change in transaction count (Q4 vs Q1)
- **Avg_Utilization_Ratio**: Average card utilization ratio

---

## 🔒 Data Privacy

This is a synthetic dataset created for educational and research purposes. It does not contain real customer data.

---

## 📥 How to Use

### If you cloned this repository:

The dataset should already be in this folder as `BankChurners.csv`.

### If the dataset is missing:

1. Download from Kaggle: https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers
2. Extract the zip file
3. Place `BankChurners.csv` in this folder

### In the Notebook:

The dataset is loaded with:
```python
df = pd.read_csv('data/BankChurners.csv')
```

Or if running from the root directory:
```python
df = pd.read_csv('BankChurners.csv')
```

---

## 🧪 Sample Data (Optional)

To create a smaller sample for quick testing:

```python
import pandas as pd

# Load full dataset
df = pd.read_csv('BankChurners.csv')

# Create sample (first 1,000 rows)
df_sample = df.head(1000)

# Save sample
df_sample.to_csv('BankChurners_sample.csv', index=False)

print(f"Sample created: {len(df_sample)} rows")
```

Then update the notebook to use `BankChurners_sample.csv` for faster testing.

---

## ✅ Verification

To verify the dataset is loaded correctly:

```python
import pandas as pd

# Load dataset
df = pd.read_csv('BankChurners.csv')

# Check shape
print(f"Shape: {df.shape}")
# Expected output: Shape: (10127, 23)

# Check columns
print(f"\nColumns: {df.columns.tolist()}")

# Check for missing values
print(f"\nMissing values: {df.isnull().sum().sum()}")
# Expected output: Missing values: 0

# Check target distribution
print(f"\nChurn distribution:")
print(df['Attrition_Flag'].value_counts())
```

Expected output:
```
Existing Customer     8500
Attrited Customer     1627
```

---

## 📊 Data Quality

- ✅ No missing values
- ✅ No duplicate records
- ✅ Consistent data types
- ✅ Realistic value ranges
- ⚠️ Class imbalance (16% churn rate)
  - Handled in the notebook using SMOTE

---

## 🎯 Usage Notes

1. **Class Imbalance**: The dataset has ~16% churn rate. We handle this using SMOTE (Synthetic Minority Over-sampling Technique) in the modeling phase.

2. **Feature Selection**: Some columns (like CLIENTNUM, Naive_Bayes predictions) are removed during preprocessing as they're not useful for modeling.

3. **Data Splitting**: We use an 80-20 train-test split with stratification to maintain class distribution.

4. **Scaling**: Numerical features are standardized using StandardScaler after splitting to prevent data leakage.

---

## 📚 References

- **Original Dataset**: [Kaggle - Credit Card Customers](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)
- **License**: Database Contents License (DbCL) v1.0
- **Citation**: Credit Card customers, Sakshi Goyal

---

**Last Updated**: February 2026
