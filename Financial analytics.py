import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib


def preprocess_data(file_path):
    df = pd.read_csv(file_path)


    print("Data Info:")
    print(df.info())


    print("Missing Values:")
    print(df.isnull().sum())

 
    df.fillna(method='ffill', inplace=True)


    df['Market Capitalization in Crores'] = pd.to_numeric(df['Market Capitalization in Crores'], errors='coerce')
    df['Quarterly Sale in crores'] = pd.to_numeric(df['Quarterly Sale in crores'], errors='coerce')


    cleaned_path = 'data/cleaned_companies.csv'
    df.to_csv(cleaned_path, index=False)
    print(f"Cleaned data saved to {cleaned_path}")


def perform_eda():
    df = pd.read_csv('data/cleaned_companies.csv')

   
    print("Summary Statistics:")
    print(df.describe())

   
    plt.figure(figsize=(12, 6))
    sns.histplot(df['Market Capitalization in Crores'], bins=30, kde=True, color='blue')
    plt.title('Distribution of Market Capitalization')
    plt.xlabel('Market Capitalization in Crores')
    plt.ylabel('Frequency')
    plt.show()

   
    plt.figure(figsize=(12, 6))
    sns.histplot(df['Quarterly Sale in crores'], bins=30, kde=True, color='green')
    plt.title('Distribution of Quarterly Sales')
    plt.xlabel('Quarterly Sale in Crores')
    plt.ylabel('Frequency')
    plt.show()

  
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='Quarterly Sale in crores', y='Market Capitalization in Crores', data=df)
    plt.title('Market Capitalization vs. Quarterly Sales')
    plt.xlabel('Quarterly Sale in Crores')
    plt.ylabel('Market Capitalization in Crores')
    plt.show()


def engineer_features():
    df = pd.read_csv('data/cleaned_companies.csv')

   
    df['Market Cap to Sales Ratio'] = df['Market Capitalization in Crores'] / df['Quarterly Sale in crores']

 
    enhanced_path = 'data/enhanced_companies.csv'
    df.to_csv(enhanced_path, index=False)
    print(f"Enhanced data saved to {enhanced_path}")


def analyze_factors():
    df = pd.read_csv('data/enhanced_companies.csv')

  
    X = df[['Quarterly Sale in crores', 'Market Cap to Sales Ratio']]
    y = df['Market Capitalization in Crores']

  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   
    model = LinearRegression()
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    print(f"R2 Score: {r2_score(y_test, y_pred)}")


    joblib.dump(model, 'model.pkl')
    print("Model saved as model.pkl")


    coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
    print("Model Coefficients:")
    print(coefficients)


def generate_report():
    df = pd.read_csv('data/enhanced_companies.csv')


    top_10_companies = df.nlargest(10, 'Market Capitalization in Crores')
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Market Capitalization in Crores', y='Name of Company', data=top_10_companies, palette='viridis')
    plt.title('Top 10 Companies by Market Capitalization')
    plt.xlabel('Market Capitalization in Crores')
    plt.ylabel('Company')
    plt.show()


    plt.figure(figsize=(10, 8))
    sns.heatmap(df[['Market Capitalization in Crores', 'Quarterly Sale in crores', 'Market Cap to Sales Ratio']].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

if __name__ == "__main__":
   
    data_file_path = 'data/companies.csv'
    
  
    preprocess_data(data_file_path)

    
    perform_eda()


    engineer_features()

   
    analyze_factors()

    
    generate_report()
