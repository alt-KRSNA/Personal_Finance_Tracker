import pandas as pd
import random
from faker import Faker

# Initialize the Faker library for random data generation
fake = Faker()

# Define the categories
categories = ['Travel', 'Food', 'Party', 'Shopping', 'Health', 'Entertainment']

# Create 100 rows of random data
data = []
for _ in range(100):
    date = fake.date_this_decade()
    amount = round(random.uniform(100, 5000), 2)
    description = fake.city()
    category = random.choice(categories)
    data.append([date, amount, description, category])

# Create the DataFrame
df = pd.DataFrame(data, columns=['Date', 'Amount', 'Description', 'Category'])

# Save the DataFrame to a CSV file
df.to_csv('expenses.csv', index=False)

print("CSV file 'expenses.csv' created successfully!")
