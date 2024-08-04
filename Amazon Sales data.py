import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from flask import Flask, request, jsonify


def preprocess_data():
   
    orders = pd.read_csv('amazon_orders.csv')
    products = pd.read_csv('amazon_products.csv')
    customers = pd.read_csv('amazon_customers.csv')

   
    print("Orders Data:")
    print(orders.info())
    print("Products Data:")
    print(products.info())
    print("Customers Data:")
    print(customers.info())

   
    orders.fillna(method='ffill', inplace=True)
    products.fillna(method='ffill', inplace=True)
    customers.fillna(method='ffill', inplace=True)

    orders['order_date'] = pd.to_datetime(orders['order_date'])

    orders.to_csv('cleaned_amazon_orders.csv', index=False)
    products.to_csv('cleaned_amazon_products.csv', index=False)
    customers.to_csv('cleaned_amazon_customers.csv', index=False)


def perform_eda():
  
    orders = pd.read_csv('cleaned_amazon_orders.csv')
    products = pd.read_csv('cleaned_amazon_products.csv')
    customers = pd.read_csv('cleaned_amazon_customers.csv')

   
    print("Orders Statistics:")
    print(orders.describe())
    print("Products Statistics:")
    print(products.describe())
    print("Customers Statistics:")
    print(customers.describe())

   
    orders.hist(bins=30, figsize=(20, 15))
    plt.show()


    orders['order_date'] = pd.to_datetime(orders['order_date'])
    sales_trend = orders.groupby(orders['order_date'].dt.to_period('M')).sum()
    sales_trend['sales_amount'].plot(figsize=(10, 6), title='Sales Trend Over Time')
    plt.show()


    top_products = orders.groupby('product_id').sum().sort_values('sales_amount', ascending=False).head(10)
    top_products = pd.merge(top_products, products, on='product_id')
    sns.barplot(x='sales_amount', y='product_name', data=top_products)
    plt.title('Top 10 Selling Products')
    plt.show()


def feature_engineering():
  
    orders = pd.read_csv('cleaned_amazon_orders.csv')
    products = pd.read_csv('cleaned_amazon_products.csv')
    customers = pd.read_csv('cleaned_amazon_customers.csv')


    orders['order_month'] = orders['order_date'].dt.month
    orders['order_year'] = orders['order_date'].dt.year

    data = pd.merge(orders, products, on='product_id')
    data = pd.merge(data, customers, on='customer_id')

 
    data.to_csv('merged_data.csv', index=False)


def train_model():
   
    data = pd.read_csv('merged_data.csv')

 
    X = data[['order_month', 'order_year', 'price', 'ratings']]
    y = data['sales_amount']

   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 
    model = LinearRegression()
    model.fit(X_train, y_train)

 
    y_pred = model.predict(X_test)
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    print(f"R2 Score: {r2_score(y_test, y_pred)}")

   
    joblib.dump(model, 'model.pkl')

   
    coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
    print("Model Coefficients:")
    print(coefficients)

  
    coefficients.plot(kind='bar')
    plt.title('Feature Coefficients')
    plt.show()


def create_flask_app():
    app = Flask(__name__)

 
    model = joblib.load('model.pkl')

    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.get_json(force=True)
        input_data = np.array([data['order_month'], data['order_year'], data['price'], data['ratings']])
        prediction = model.predict([input_data])
        return jsonify({'predicted_sales_amount': prediction[0]})

    if __name__ == '__main__':
        app.run(debug=True)


if __name__ == "__main__":
    preprocess_data()         
    perform_eda()             
    feature_engineering()     
    train_model()            
   
    # create_flask_app()     
