from flask import Flask, jsonify, render_template
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load traffic data from Excel/CSV
file_path = "DelhiTrafficDensityDataset/DelhiTrafficDensityDataset/Dec15.csv"
df = pd.read_excel(file_path) if file_path.endswith(".xlsx") else pd.read_csv(file_path)

# Train the Linear Regression model
features = ['time', 'day_of_week']
target = 'traffic_volume'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

def get_real_time_data():
    """Simulate fetching real-time traffic and weather data"""
    now = datetime.datetime.now()
    current_time = now.hour + now.minute / 60  # Convert to decimal format
    current_day = now.weekday()  # Monday = 0, ..., Sunday = 6

    # Simulated weather and traffic data (Replace with real API)
    weather_data = {
        "temperature": 25,
        "humidity": 60,
        "weather_condition": "Clear"
    }
    
    traffic_data = {
        "traffic_count": 120
    }

    return current_time, current_day, weather_data, traffic_data

@app.route('/')
def index():
    """Serve the HTML frontend"""
    return render_template("index.html")

@app.route('/data', methods=['GET'])
def get_prediction():
    """API to return real-time traffic prediction"""
    current_time, current_day, weather_data, traffic_data = get_real_time_data()
    
    input_data = pd.DataFrame({'time': [current_time], 'day_of_week': [current_day]})
    predicted_traffic = model.predict(input_data)[0]

    return jsonify({
        "traffic": traffic_data,
        "weather": weather_data,
        "predicted_traffic": predicted_traffic
    })

if __name__ == '__main__':
    app.run(debug=True)
