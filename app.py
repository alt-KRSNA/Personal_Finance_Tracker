from flask import Flask, render_template, request, redirect
import pandas as pd
import os
from datetime import datetime
from utils import train_category_model, train_spending_model, predict_next_spending


app = Flask(__name__)
CSV_FILE = 'expense.csv'

# Save new expense
def save_expense(date, amount, desc, category='Other', csv_path=CSV_FILE):
    data = {'Date': [date], 'Amount': [amount], 'Description': [desc], 'Category': [category]}
    df = pd.DataFrame(data)
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)

# Get current stats
def get_expense_stats(csv_path=CSV_FILE):
    if not os.path.exists(csv_path) or os.stat(csv_path).st_size == 0:
        return {
            'current_month_total': 0,
            'last_month_total': 0,
            'avg_monthly': 0,
            'total_expenses': 0
        }

    df = pd.read_csv(csv_path)

    # Normalize and convert columns
    df.columns = df.columns.str.lower()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df = df.dropna(subset=['date', 'amount'])

    now = datetime.now()
    current_month = now.month
    current_year = now.year

    current_month_exp = df[(df['date'].dt.month == current_month) & (df['date'].dt.year == current_year)]
    last_month = current_month - 1 if current_month > 1 else 12
    last_month_year = current_year if current_month > 1 else current_year - 1
    last_month_exp = df[(df['date'].dt.month == last_month) & (df['date'].dt.year == last_month_year)]

    df['month'] = df['date'].dt.to_period('M')
    avg_monthly = df.groupby('month')['amount'].sum().mean()

    return {
        'current_month_total': current_month_exp['amount'].sum(),
        'last_month_total': last_month_exp['amount'].sum(),
        'avg_monthly': avg_monthly,
        'total_expenses': df['amount'].sum()
    }

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        prediction = predict_next_spending(CSV_FILE)

    stats = get_expense_stats(CSV_FILE)
    return render_template('index.html', prediction=prediction, stats=stats)

@app.route('/add', methods=['POST'])
def add():
    date = request.form['date']
    amount = float(request.form['amount'])
    desc = request.form['description']
    save_expense(date, amount, desc)
    return redirect('/')

@app.route('/view')
def view():
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        records = df.to_dict(orient='records')
    else:
        records = []
    return render_template('view.html', records=records)

@app.route('/train_category')
def train_category():
    train_category_model(CSV_FILE)
    return redirect('/')

@app.route('/train_spending')
def train_spending():
    train_spending_model(CSV_FILE)
    return redirect('/')

@app.route('/predict_spending')
def predict_spending_route():
    prediction = predict_next_spending(CSV_FILE)
    stats = get_expense_stats(CSV_FILE)
    return render_template('index.html', prediction=prediction, stats=stats)

if __name__ == '__main__':
    app.run(debug=True)
