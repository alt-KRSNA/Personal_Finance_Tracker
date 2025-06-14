import os
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
# File paths
EXPENSES_FILE = "expenses.csv"
CATEGORY_MODEL_PATH = os.path.join("models", "expense_classifier.pkl")
SPENDING_MODEL_PATH = os.path.join("models", "spending_predictor.pkl")


def load_expenses():
    """Load expenses from CSV or return empty DataFrame with correct structure."""
    if os.path.exists(EXPENSES_FILE):
        df = pd.read_csv(EXPENSES_FILE)
    else:
        df = pd.DataFrame(columns=["Date", "Amount", "Description", "Category"])
    return df


def save_expenses(df):
    """Save the DataFrame to CSV."""
    df.to_csv(EXPENSES_FILE, index=False)


def predict_category(description):
    """Predict the expense category using the trained model."""
    if not os.path.exists(CATEGORY_MODEL_PATH):
        return "Unknown"
    with open(CATEGORY_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model.predict([description])[0]


def train_category_model(df):
    """Train a category classification model based on expense descriptions."""
    df = df[df["Category"].str.lower() != "unknown"]
    if df.empty:
        return "No data to train on."

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(df["Description"], df["Category"])

    with open(CATEGORY_MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    return "✅ Category prediction model trained successfully."


def train_spending_model(df):
    """Train a simple Linear Regression model to predict monthly spending."""
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    monthly = df.groupby(df["Date"].dt.to_period("M"))["Amount"].sum().reset_index()
    monthly["Index"] = range(len(monthly))

    if monthly.empty:
        return "No valid data for training spending model."

    model = LinearRegression()
    X = monthly[["Index"]].values
    y = monthly["Amount"].values
    model.fit(X, y)

    with open(SPENDING_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return "✅ Spending prediction model trained successfully."


def predict_next_spending(df):
    """Predict next month's spending using the trained Linear Regression model."""
    if not os.path.exists(SPENDING_MODEL_PATH):
        return 0.0

    with open(SPENDING_MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    monthly = df.groupby(df["Date"].dt.to_period("M"))["Amount"].sum().reset_index()
    next_index = len(monthly)

    try:
        prediction = model.predict(np.array([[next_index]]))[0]
        return round(float(prediction), 2)
    except Exception as e:
        print(f"Prediction error: {e}")
        return 0.0
def get_expense_stats(csv_path='C:/Users/krish/Downloads/Hackathon1/expenses.csv'):
    if not os.path.exists(csv_path) or os.stat(csv_path).st_size == 0:
        return {
            'current_month_total': 0,
            'last_month_total': 0,
            'avg_monthly': 0,
            'total_expenses': 0
        }

    df = pd.read_csv(csv_path)

    # Normalize column names
    df.columns = df.columns.str.lower()

    # Convert types
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df = df.dropna(subset=['date', 'amount'])

    now = datetime.now()
    current_month = now.month
    current_year = now.year

    # Current month total
    current_month_exp = df[(df['date'].dt.month == current_month) & (df['date'].dt.year == current_year)]
    current_month_total = current_month_exp['amount'].sum()

    # Last month total
    last_month = current_month - 1 if current_month > 1 else 12
    last_month_year = current_year if current_month > 1 else current_year - 1
    last_month_exp = df[(df['date'].dt.month == last_month) & (df['date'].dt.year == last_month_year)]
    last_month_total = last_month_exp['amount'].sum()

    # Average monthly total
    df['month'] = df['date'].dt.to_period('M')
    monthly_avg = df.groupby('month')['amount'].sum().mean()

    total_expenses = df['amount'].sum()

    return {
        'current_month_total': current_month_total,
        'last_month_total': last_month_total,
        'avg_monthly': monthly_avg,
        'total_expenses': total_expenses
    }
