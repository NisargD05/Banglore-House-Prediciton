#  Bangalore House Price Prediction

An end-to-end machine learning project that predicts house prices in Bangalore based on property features such as location, total square footage, number of bedrooms, bathrooms, and balconies.  
The final model is deployed as an interactive web application using Streamlit.

---

##  Project Overview

- Cleaned and preprocessed real-world Bangalore housing data
- Performed feature engineering and outlier removal
- Trained and compared multiple regression models
- Selected Ridge Regression as the final model
- Built and deployed a Streamlit-based web UI for real-time predictions

---

## Machine Learning Pipeline

1. **Data Cleaning**
   - Handled missing values
   - Converted `size` to `BHK`
   - Cleaned `total_sqft` (ranges and invalid entries)
   - Removed unrealistic sqft-per-BHK entries

2. **Feature Engineering**
   - Created numerical features (BHK, bathrooms, balconies)
   - Grouped rare locations
   - One-hot encoded location feature
   - Scaled features using `StandardScaler`

3. **Modeling**
   - Linear Regression
   - Ridge Regression  (final model)
   - Lasso Regression
   - Random Forest (for comparison)

4. **Evaluation**
   - Train-test split
   - R² score comparison
   - Hyperparameter tuning using cross-validation

---

##  Deployment

- Built an interactive UI using **Streamlit**
- Saved trained model, scaler, and feature columns
- Deployed on **Streamlit Community Cloud**
- Users can input property details and get instant price predictions

---

##  Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit

---

##  Final Model Performance

- **Model:** Ridge Regression  
- **R² Score:** ~0.90 on test data  

---

##  How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app1.py
