# 🚀 Quick Start Guide - Bank Customer Churn Prediction

## Prerequisites
- Python 3.8 or higher installed
- Basic understanding of Jupyter Notebooks
- Git installed on your machine

## Step-by-Step Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/bank-churn-prediction.git
cd bank-churn-prediction
```

### 2️⃣ Create Virtual Environment (Recommended)

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Download the Dataset

**Option A: From Kaggle**
1. Go to: https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers
2. Download `BankChurners.csv`
3. Create a `data/` folder in the project root
4. Place the CSV file in the `data/` folder

**Option B: Use Your Own Data**
- Ensure your dataset has similar columns
- Update the file path in the notebook accordingly

### 5️⃣ Launch Jupyter Notebook
```bash
jupyter notebook
```

### 6️⃣ Open the Main Notebook
- Navigate to `Bank_Customer_Churn_Prediction_Enhanced.ipynb`
- Click to open
- Run cells sequentially (Shift + Enter)

## 📊 What to Expect

The notebook will guide you through:

1. **Data Loading** (~30 seconds)
2. **EDA & Visualization** (~2-3 minutes)
3. **Feature Engineering** (~1 minute)
4. **Model Training** (~3-5 minutes)
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
   - XGBoost
5. **Evaluation & Insights** (~1 minute)

**Total Runtime**: ~10-15 minutes on a standard laptop

## 🎯 Key Outputs

After running the complete notebook, you'll have:

✅ **15+ Visualizations** showing customer behavior patterns
✅ **4 Trained Models** with performance comparisons
✅ **Feature Importance Rankings** 
✅ **Saved Model Files** in the project directory:
   - `churn_model_random_forest.pkl`
   - `scaler.pkl`
   - `label_encoders.pkl`
   - `feature_names.pkl`

## 🔍 Understanding the Results

### Model Performance
Look for these metrics in the output:
- **Accuracy**: ~95% (overall correctness)
- **Recall**: ~84% (% of churners identified)
- **Precision**: ~85% (accuracy of churn predictions)
- **ROC-AUC**: ~0.97 (model discrimination ability)

### Business Insights
The notebook provides:
- Top factors causing churn
- Customer segments at highest risk
- Recommended retention strategies
- Expected financial impact

## 🛠️ Troubleshooting

### Issue: Module Not Found
**Solution**: 
```bash
pip install --upgrade -r requirements.txt
```

### Issue: Kernel Keeps Dying
**Solution**: Reduce data size or increase memory allocation
```python
# In the notebook, add this at the top
import os
os.environ['OMP_NUM_THREADS'] = '1'
```

### Issue: XGBoost Installation Fails
**Solution**: 
```bash
pip install xgboost --no-cache-dir
```
Or comment out XGBoost sections in the notebook

### Issue: Dataset Not Found
**Solution**: Update the file path in cell 2:
```python
# Change this line to your dataset location
df = pd.read_csv('YOUR_PATH/BankChurners.csv')
```

## 📝 Customization Tips

### Use Your Own Data
Replace the dataset but ensure these columns exist:
- `Attrition_Flag` (target variable)
- Customer demographics
- Transaction metrics
- Account information

### Adjust Model Parameters
Find this section in the notebook:
```python
rf_model = RandomForestClassifier(
    n_estimators=100,    # Try 200 for better accuracy
    max_depth=None,      # Try 10-20 to prevent overfitting
    random_state=42
)
```

### Change Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,       # Change to 0.3 for smaller training set
    random_state=42
)
```

## 🎓 Next Steps

After completing the notebook:

1. **Deploy the Model**
   - Build a Flask/FastAPI web service
   - Create a Streamlit dashboard
   - Integrate with your CRM system

2. **Enhance the Analysis**
   - Try deep learning models (Neural Networks)
   - Implement time-series forecasting
   - Add customer lifetime value (CLV) prediction

3. **Business Implementation**
   - Present findings to stakeholders
   - Design retention campaigns
   - Set up automated monitoring

## 🤝 Need Help?

- Check the [Main README](README.md) for detailed documentation
- Open an issue on GitHub
- Review the notebook comments for explanations
- Check scikit-learn documentation: https://scikit-learn.org/

## 📚 Learning Resources

- **Machine Learning**: [Coursera ML Course](https://www.coursera.org/learn/machine-learning)
- **Imbalanced Data**: [imbalanced-learn docs](https://imbalanced-learn.org/)
- **Feature Engineering**: [Feature Engineering Book](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)

---

**Happy Learning! 🚀**

If you find this project helpful, please consider giving it a ⭐ on GitHub!
