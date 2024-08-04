import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from flask import Flask, request, jsonify


def preprocess_data():
    world_cups = pd.read_csv('data/world_cups.csv')
    matches = pd.read_csv('data/world_cup_matches.csv')

 
    print("World Cups Data Info:")
    print(world_cups.info())
    print("Matches Data Info:")
    print(matches.info())

   
    world_cups.fillna(method='ffill', inplace=True)
    matches.fillna(method='ffill', inplace=True)

  
    world_cups.to_csv('data/cleaned_world_cups.csv', index=False)
    matches.to_csv('data/cleaned_world_cup_matches.csv', index=False)

def perform_eda():
    world_cups = pd.read_csv('data/cleaned_world_cups.csv')
    matches = pd.read_csv('data/cleaned_world_cup_matches.csv')

   
    wins_by_country = world_cups['Winner'].value_counts()
    plt.figure(figsize=(12, 8))
    sns.barplot(x=wins_by_country.index, y=wins_by_country.values, palette='viridis')
    plt.title('Number of World Cups Won by Country')
    plt.xlabel('Country')
    plt.ylabel('Number of Wins')
    plt.xticks(rotation=90)
    plt.show()

   
    world_cups['winner_goals'] = world_cups['Winner'].apply(
        lambda x: matches.query(f"team == '{x}'")['total_goals'].mean()
    )
    world_cups['runner_up_goals'] = world_cups['Runner-Up'].apply(
        lambda x: matches.query(f"team == '{x}'")['total_goals'].mean()
    )
    
    print(f"Average Goals Scored by Winners: {world_cups['winner_goals'].mean()}")
    print(f"Average Goals Scored by Runners-Up: {world_cups['runner_up_goals'].mean()}")


    host_performance = world_cups.groupby('Host Country')['Winner'].apply(
        lambda x: (x == world_cups['Host Country']).sum()
    )
    plt.figure(figsize=(12, 8))
    sns.barplot(x=host_performance.index, y=host_performance.values, palette='coolwarm')
    plt.title('Performance of Host Countries')
    plt.xlabel('Host Country')
    plt.ylabel('Number of Wins')
    plt.xticks(rotation=90)
    plt.show()


def engineer_features():
    matches = pd.read_csv('data/cleaned_world_cup_matches.csv')

  
    matches['total_goals'] = matches['home_score'] + matches['away_score']
    matches['goal_difference'] = matches['home_score'] - matches['away_score']
    
    
    matches.to_csv('data/enhanced_matches.csv', index=False)


def train_model():
    matches = pd.read_csv('data/enhanced_matches.csv')


    X = matches[['total_goals', 'goal_difference']]
    y = (matches['home_score'] > matches['away_score']).astype(int)  

   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

 
    y_pred = model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")

  
    joblib.dump(model, 'model.pkl')


def run_flask_app():
    app = Flask(__name__)
    model = joblib.load('model.pkl')

    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.get_json(force=True)
        input_features = np.array([data['total_goals'], data['goal_difference']])
        prediction = model.predict([input_features])
        return jsonify({'predicted_winner': int(prediction[0])})

    app.run(debug=True)


if __name__ == "__main__":
  
    preprocess_data()
    perform_eda()
    engineer_features()
    train_model()

  
    run_flask_app()
