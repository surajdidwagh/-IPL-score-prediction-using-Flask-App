from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('forest_model.pkl')

# Teams list
teams = ['Chennai Super Kings', 'Delhi Daredevils', 'Kings XI Punjab', 'Kolkata Knight Riders',
         'Mumbai Indians', 'Rajasthan Royals', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad']

# Encoding function
def encode_team(team):
    return [1 if team == t else 0 for t in teams]

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        batting_team = data['batting_team']
        bowling_team = data['bowling_team']
        overs = float(data['overs'])
        runs = int(data['runs'])
        wickets = int(data['wickets'])
        runs_last_5 = int(data['runs_last_5'])
        wickets_last_5 = int(data['wickets_last_5'])

        input_vector = encode_team(batting_team) + encode_team(bowling_team) + [
            runs, wickets, overs, runs_last_5, wickets_last_5
        ]

        prediction = model.predict([input_vector])[0]
        return jsonify({'predicted_score': int(round(prediction))})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
