import os
from flask import Flask, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

# Create the 'static' folder if it doesn't exist (for saving images)
if not os.path.exists('static'):
    os.makedirs('static')

# Load the data
file_path = 'Traffic.csv'  # Update with the correct path if needed
traffic_data = pd.read_csv(file_path)

# Convert 'Time' to datetime and extract the hour, minute
traffic_data['Time'] = pd.to_datetime(traffic_data['Time'], format='%H:%M:%S')
traffic_data['Hour'] = traffic_data['Time'].dt.hour
traffic_data['Minute'] = traffic_data['Time'].dt.minute

# Add the 'Traffic Situation' column based on 'Total' traffic levels
traffic_data['Traffic Situation'] = np.where(traffic_data['Total'] > traffic_data['Total'].mean(), 'High', 'Low')

# One-hot encode categorical features (weather, holiday, location, day_of_week)
encoder = OneHotEncoder(drop='first', sparse_output=False)  # Ensure 'sparse_output=False' is set
categorical_columns = ['weather', 'holiday', 'location', 'day_of_week']

# Encoding categorical columns and adding them back to the dataframe
encoded_features = pd.DataFrame(encoder.fit_transform(traffic_data[categorical_columns]))
encoded_features.columns = encoder.get_feature_names_out(categorical_columns)
traffic_data = pd.concat([traffic_data, encoded_features], axis=1)

# Prepare features and target
X = traffic_data[['Hour', 'Minute', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount'] + list(encoded_features.columns)]
y = traffic_data['Total']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the entire data
traffic_data['Predicted_Total'] = rf_model.predict(X)

# Function to save Traffic Prediction vs Actual plot
def save_traffic_plot():
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=traffic_data[:500], x='Time', y='Total', label='Actual Total Traffic', marker='o')
    sns.lineplot(data=traffic_data[:500], x='Time', y='Predicted_Total', label='Predicted Total Traffic', marker='x')
    plt.xticks(rotation=90)
    plt.title('Traffic Prediction vs Actual')
    plt.xlabel('Time')
    plt.ylabel('Total Traffic')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('static/traffic_prediction_plot.png')  # Save the plot as an image
    plt.close()

# Function to save Traffic Situation Distribution plot
def save_traffic_situation_plot():
    plt.figure(figsize=(12, 6))
    sns.countplot(data=traffic_data, x='Traffic Situation', order=traffic_data['Traffic Situation'].value_counts().index)
    plt.title('Traffic Situation Distribution')
    plt.xlabel('Traffic Situation')
    plt.ylabel('Count')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('static/traffic_situation_plot.png')  # Save the plot as an image
    plt.close()

# Function to save vehicle counts by hour plots (stacked bar charts)
def save_vehicle_counts_by_hour():
    unique_days = traffic_data['day_of_week'].unique()
    for day in unique_days:
        day_data = traffic_data[traffic_data['day_of_week'] == day]
        hourly_vehicle_counts = day_data.groupby('Hour')[['CarCount', 'BikeCount', 'BusCount', 'TruckCount']].sum()
        plt.figure(figsize=(12, 6))
        hourly_vehicle_counts.plot(kind='bar', stacked=True, color=['blue', 'green', 'orange', 'red'], edgecolor='black')
        plt.title(f"Vehicle Counts by Hour: {day}", fontsize=16)
        plt.xlabel("Hour of the Day", fontsize=14)
        plt.ylabel("Number of Vehicles", fontsize=14)
        plt.xticks(rotation=0)
        plt.legend(title="Vehicle Type", fontsize=12)
        plt.grid(axis='y', linestyle='--', linewidth=0.7)
        plt.tight_layout()
        plt.savefig(f'static/vehicle_counts_by_hour_{day}.png')  # Save the plot as an image
        plt.close()

# Function to save the average traffic per hour plot
def save_avg_traffic_by_hour():
    unique_days = traffic_data['day_of_week'].unique()
    for day in unique_days:
        day_data = traffic_data[traffic_data['day_of_week'] == day]
        hourly_avg_traffic = day_data.groupby('Hour')['Total'].mean()
        plt.figure(figsize=(12, 6))
        hourly_avg_traffic.plot(kind='line', marker='o', linestyle='-', color='purple', linewidth=2)
        plt.title(f"Average Traffic per Hour: {day}", fontsize=16)
        plt.xlabel("Hour of the Day", fontsize=14)
        plt.ylabel("Average Total Traffic", fontsize=14)
        plt.grid(axis='both', linestyle='--', linewidth=0.7)
        plt.tight_layout()
        plt.savefig(f'static/avg_traffic_by_hour_{day}.png')  # Save the plot as an image
        plt.close()

# Call the functions to save the plots
save_traffic_plot()
save_traffic_situation_plot()
save_vehicle_counts_by_hour()
save_avg_traffic_by_hour()

# Print model evaluation
mse = mean_squared_error(y_test, rf_model.predict(X_test))
print(f"Mean Squared Error on Test Data: {mse:.2f}")

@app.route('/')
def index():
    # Render the HTML template and pass the plot image paths
    return render_template('index.html', 
                           traffic_plot='static/traffic_prediction_plot.png',
                           situation_plot='static/traffic_situation_plot.png',
                           vehicle_counts_plots=[f'static/vehicle_counts_by_hour_{day}.png' for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']],
                           avg_traffic_plots=[f'static/avg_traffic_by_hour_{day}.png' for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']])

if __name__ == '__main__':
    app.run(debug=True)
