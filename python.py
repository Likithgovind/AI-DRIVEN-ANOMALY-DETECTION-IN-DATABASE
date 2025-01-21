import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import BytesIO
import os

# Suppress interactive backend warnings
import matplotlib
matplotlib.use('Agg')

def get_user_database():
    """Prompt the user to provide a database file path or URL."""
    file_input = input("Enter the path to your database file (local path or URL): ").strip('"')
    try:
        if file_input.startswith("http://") or file_input.startswith("https://"):
            # Handle file from URL
            response = requests.get(file_input)
            if response.status_code != 200:
                raise ValueError("Failed to download the file. Please check the URL.")
            file_format = input("Enter the file format ('csv' or 'excel'): ").lower()
            if file_format == 'csv':
                try:
                    data = pd.read_csv(BytesIO(response.content))
                except UnicodeDecodeError:
                    print("utf-8 decoding failed. Trying 'latin1' encoding.")
                    data = pd.read_csv(BytesIO(response.content), encoding='latin1')
            elif file_format == 'excel':
                data = pd.read_excel(BytesIO(response.content))
            else:
                raise ValueError("Unsupported file format. Please specify 'csv' or 'excel'.")
        else:
            # Handle local file
            if not os.path.exists(file_input):
                raise ValueError("File not found. Please provide a valid file path.")
            file_size_mb = os.path.getsize(file_input) / (1024 * 1024)
            if file_size_mb > 100:
                raise ValueError("File size exceeds 100 MB limit.")
            if file_input.endswith('.csv'):
                try:
                    data = pd.read_csv(file_input)
                except UnicodeDecodeError:
                    print("utf-8 decoding failed. Trying 'latin1' encoding.")
                    data = pd.read_csv(file_input, encoding='latin1')
            elif file_input.endswith('.xlsx') or file_input.endswith('.xls'):
                data = pd.read_excel(file_input)
            else:
                raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
        print("Database loaded successfully.")
        return data
    except Exception as e:
        print(f"Error loading database: {e}")
        return None

def preprocess_data(data):
    """Preprocess the data: handle missing values and scale features."""
    numeric_data = data.select_dtypes(include=[np.number])
    if numeric_data.empty:
        raise ValueError("No numeric data available for preprocessing.")

    # Handle missing values by imputing with the column mean
    numeric_data = numeric_data.fillna(numeric_data.mean())

    # Drop columns with all NaN values (if any remain)
    numeric_data = numeric_data.dropna(axis=1, how='all')

    if numeric_data.empty:
        raise ValueError("All numeric columns have NaN values. Unable to process data.")

    # Scale features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)

    # Combine scaled numeric data with original non-numeric columns
    non_numeric_data = data.select_dtypes(exclude=[np.number])
    processed_data = pd.concat([non_numeric_data.reset_index(drop=True), 
                                 pd.DataFrame(scaled_data, columns=numeric_data.columns)], axis=1)
    return processed_data

def detect_anomalies(data):
    """Detect anomalies in the database using Isolation Forest."""
    numeric_data = data.select_dtypes(include=[np.number])
    if numeric_data.empty:
        raise ValueError("No numeric data available for anomaly detection.")

    # Ensure no NaN values remain in the numeric columns
    if numeric_data.isna().any().any():
        raise ValueError("Processed data still contains NaN values.")

    # Initialize and fit the Isolation Forest model
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(numeric_data)

    # Predict anomalies (-1 indicates anomaly, 1 indicates normal)
    anomaly_labels = model.predict(numeric_data)
    data['anomaly'] = anomaly_labels

    # Calculate anomaly percentage
    anomaly_percentage = (anomaly_labels == -1).mean() * 100

    # Separate anomalies and normal data
    anomalies = data[data['anomaly'] == -1]
    normal_data = data[data['anomaly'] == 1]

    print(f"Anomalies detected: {len(anomalies)} ({anomaly_percentage:.2f}%)")
    return anomalies, normal_data, anomaly_percentage

def add_features(data):
    """Add a few synthetic features to enhance analysis."""
    numeric_data = data.select_dtypes(include=[np.number])
    if numeric_data.empty:
        raise ValueError("No numeric data found to compute features.")

    data['feature_sum'] = numeric_data.sum(axis=1)
    data['feature_mean'] = numeric_data.mean(axis=1)
    data['feature_std'] = numeric_data.std(axis=1)

    # Ensure all column names are strings for safe handling in visualizations
    data.columns = [str(col) for col in data.columns]
    print("New features added successfully.")
    return data

def plot_scatter(data):
    """Plot scatter visualization for anomalies."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data.index, y=data['feature_sum'], hue=data['anomaly'], palette=['red', 'green'], alpha=0.7)
    plt.title('Scatter Plot of Anomalies')
    plt.xlabel('Index')
    plt.ylabel('Feature Sum')
    plt.legend(title='Anomaly', loc='upper right', labels=['Anomaly (-1)', 'Normal (1)'])
    plt.savefig("scatter_plot.png")  # Save plot to file
    print("Scatter plot saved as 'scatter_plot.png'.")
    plt.close()

def plot_anomaly_bar_chart(anomalies, normal_data, anomaly_percentage):
    """Plot a bar chart showing counts of anomalies and normal data."""
    sizes = [len(anomalies), len(normal_data)]
    labels = ['Anomalies', 'Normal Data']
    plt.figure(figsize=(8, 6))
    sns.barplot(x=labels, y=sizes, palette=['red', 'green'])
    plt.title(f'Counts of Anomalies and Normal Data (Anomalies: {anomaly_percentage:.2f}%)')
    plt.ylabel('Count')
    plt.savefig("anomaly_bar_chart.png")  # Save plot to file
    print("Anomaly bar chart saved as 'anomaly_bar_chart.png'.")
    plt.close()

if __name__ == "__main__":
    # Step 1: Load user database
    user_data = get_user_database()
    if user_data is not None:
        print("Original Data:")
        print(user_data.head())

        try:
            # Step 2: Preprocess the data
            processed_data = preprocess_data(user_data)

            # Ensure no NaN values remain in the numeric columns before anomaly detection
            if processed_data.isna().any().any():
                raise ValueError("Processed data still contains NaN values.")

            # Step 3: Add new features
            enhanced_data = add_features(processed_data)

            # Step 4: Detect anomalies
            anomalies, normal_data, anomaly_percentage = detect_anomalies(enhanced_data)

            # Step 5: Save and display results
            enhanced_data.to_csv("enhanced_database.csv", index=False)
            anomalies.to_csv("anomalies.csv", index=False)
            normal_data.to_csv("normal_data.csv", index=False)

            print("\nAnomaly detection complete.")
            print("Enhanced database saved as 'enhanced_database.csv'.")
            print("Anomalies saved as 'anomalies.csv'.")
            print("Normal data saved as 'normal_data.csv'.")

            # Step 6: Visualizations
            plot_scatter(enhanced_data)
            plot_anomaly_bar_chart(anomalies, normal_data, anomaly_percentage)
        except Exception as e:
            print(f"Error during processing: {e}")
