import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import requests
import json
import os
import pywt
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

def fetch_emsc_data():
    """Fetch earthquake data from EMSC API with a larger date range"""
    url = "https://www.seismicportal.eu/fdsnws/event/1/query"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2*365)
    
    params = {
        "start": start_date.strftime("%Y-%m-%d"),
        "end": end_date.strftime("%Y-%m-%d"),
        "minmag": 4.0,
        "format": "json" 
    }
    
    try:
        print(f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        if not response.text:
            raise ValueError("Empty response from server")
            
        data = response.json()
        
        if not data or 'features' not in data:
            raise ValueError("Invalid data format received from server")
            
        print(f"Successfully fetched {len(data['features'])} earthquake records")
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {str(e)}")
        print("Falling back to sample data...")
        return {
            'features': [
                {
                    'properties': {
                        'mag': 5.0,
                        'depth': 10.0,
                        'time': '2023-01-01T00:00:00Z'
                    },
                    'geometry': {
                        'coordinates': [0.0, 0.0, 0.0]
                    }
                }
            ]
        }
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {str(e)}")
        print("Falling back to sample data...")
        return {
            'features': [
                {
                    'properties': {
                        'mag': 5.0,
                        'depth': 10.0,
                        'time': '2023-01-01T00:00:00Z'
                    },
                    'geometry': {
                        'coordinates': [0.0, 0.0, 0.0]
                    }
                }
            ]
        }

def preprocess_data(data):
    """Preprocess the earthquake data with additional features"""
    df = pd.DataFrame(data['features'])
    
    df['magnitude'] = df['properties'].apply(lambda x: x['mag'])
    df['depth'] = df['properties'].apply(lambda x: x['depth'])
    df['timestamp'] = df['properties'].apply(lambda x: x['time'])
    df['latitude'] = df['geometry'].apply(lambda x: x['coordinates'][1])
    df['longitude'] = df['geometry'].apply(lambda x: x['coordinates'][0])
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    df['hour'] = df['timestamp'].dt.hour
    
    return df

def descriptive_statistics(df):
    """Generate descriptive statistics"""
    stats = {
        'magnitude_stats': df['magnitude'].describe(),
        'depth_stats': df['depth'].describe(),
        'total_earthquakes': len(df),
        'date_range': [df['timestamp'].min(), df['timestamp'].max()]
    }
    return stats

def time_series_analysis(df):
    """Perform time series analysis"""
    # Group by month and count earthquakes
    monthly_counts = df.groupby(df['timestamp'].dt.to_period('M')).size()
    
    # Calculate magnitude trends
    monthly_avg_magnitude = df.groupby(df['timestamp'].dt.to_period('M'))['magnitude'].mean()
    
    return {
        'monthly_counts': monthly_counts,
        'monthly_avg_magnitude': monthly_avg_magnitude
    }

def wavelet_analysis(df):
    """Perform wavelet analysis on the time series data"""
    # Prepare time series data
    daily_counts = df.groupby(df['timestamp'].dt.date).size()
    daily_counts = daily_counts.reindex(pd.date_range(daily_counts.index.min(), daily_counts.index.max()))
    daily_counts = daily_counts.fillna(0)
    
    # Perform wavelet transform
    wavelet = 'db4'
    coeffs = pywt.wavedec(daily_counts.values, wavelet, level=4)
    
    # Reconstruct signals for different frequency bands
    reconstructed_signals = []
    for i in range(len(coeffs)):
        coeffs_copy = coeffs.copy()
        for j in range(len(coeffs)):
            if j != i:
                coeffs_copy[j] = np.zeros_like(coeffs[j])
        reconstructed = pywt.waverec(coeffs_copy, wavelet)
        reconstructed_signals.append(reconstructed[:len(daily_counts)])
    
    # Create wavelet analysis plots
    plt.figure(figsize=(15, 12))
    
    # Original time series
    plt.subplot(4, 1, 1)
    plt.plot(daily_counts.index, daily_counts.values)
    plt.title('Original Daily Earthquake Counts')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.grid(True)
    
    # Wavelet decomposition levels
    for i in range(3):
        plt.subplot(4, 1, i+2)
        plt.plot(daily_counts.index, reconstructed_signals[i+1])
        plt.title(f'Wavelet Decomposition Level {i+1}')
        plt.xlabel('Date')
        plt.ylabel('Amplitude')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('output/wavelet_analysis.png')
    plt.close()
    
    # Create power spectrum plot
    plt.figure(figsize=(12, 6))
    for i in range(1, len(coeffs)):
        plt.plot(np.abs(coeffs[i]), label=f'Level {i}')
    plt.title('Wavelet Power Spectrum')
    plt.xlabel('Coefficient Index')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)
    plt.savefig('output/wavelet_power_spectrum.png')
    plt.close()
    
    return {
        'daily_counts': daily_counts,
        'reconstructed_signals': reconstructed_signals,
        'wavelet': wavelet,
        'coefficients': coeffs
    }

def clustering_analysis(df):
    """Perform clustering analysis with accuracy metrics"""
    # Prepare features for clustering
    features = df[['latitude', 'longitude', 'depth', 'magnitude']].copy()
    
    # Add more sophisticated derived features
    features['magnitude_squared'] = features['magnitude'] ** 2
    features['depth_normalized'] = features['depth'] / features['depth'].max()
    features['magnitude_depth_ratio'] = features['magnitude'] / (features['depth'] + 1)
    
    # Add temporal features
    features['day_of_year'] = df['timestamp'].dt.dayofyear
    features['hour'] = df['timestamp'].dt.hour
    
    # Add geographic features
    features['distance_from_equator'] = abs(features['latitude'])
    features['distance_from_prime_meridian'] = abs(features['longitude'])
    
    # Handle missing values
    features = features.fillna(features.mean())
    
    # Normalize features using MinMaxScaler
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=0.95)  # Keep 95% of variance
    pca_features = pca.fit_transform(scaled_features)
    
    # Try DBSCAN clustering
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    labels = dbscan.fit_predict(pca_features)
    
    # If DBSCAN doesn't find enough clusters, fall back to KMeans
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters < 2:
        print("DBSCAN didn't find enough clusters, falling back to KMeans...")
        # Find optimal number of clusters using silhouette score
        silhouette_scores = []
        K = range(2, 11)
        
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
            labels = kmeans.fit_predict(pca_features)
            silhouette_scores.append(silhouette_score(pca_features, labels))
        
        optimal_k = K[np.argmax(silhouette_scores)]
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
        labels = kmeans.fit_predict(pca_features)
        n_clusters = optimal_k
    
    # Assign clusters back to the dataframe
    df['cluster'] = labels
    
    # Calculate cluster statistics
    cluster_stats = df.groupby('cluster').agg({
        'magnitude': ['mean', 'std', 'count'],
        'depth': ['mean', 'std'],
        'latitude': ['mean', 'std'],
        'longitude': ['mean', 'std']
    }).round(2)
    
    # Create cluster visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Magnitude vs Depth colored by cluster
    plt.subplot(2, 2, 1)
    scatter = plt.scatter(df['magnitude'], df['depth'], c=df['cluster'], cmap='viridis')
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('Magnitude')
    plt.ylabel('Depth')
    plt.title('Magnitude vs Depth by Cluster')
    plt.grid(True)
    
    # Plot 2: Latitude vs Longitude colored by cluster
    plt.subplot(2, 2, 2)
    scatter = plt.scatter(df['longitude'], df['latitude'], c=df['cluster'], cmap='viridis')
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Geographic Distribution by Cluster')
    plt.grid(True)
    
    # Plot 3: PCA components
    plt.subplot(2, 2, 3)
    scatter = plt.scatter(pca_features[:, 0], pca_features[:, 1], c=df['cluster'], cmap='viridis')
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA Projection of Clusters')
    plt.grid(True)
    
    # Plot 4: Explained variance ratio
    plt.subplot(2, 2, 4)
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('output/clustering_analysis.png')
    plt.close()
    
    # Calculate final metrics
    silhouette_avg = silhouette_score(pca_features, labels)
    calinski_avg = calinski_harabasz_score(pca_features, labels)
    
    print(f"Number of clusters: {n_clusters}")
    print(f"Average Silhouette Score: {silhouette_avg:.3f}")
    print(f"Calinski-Harabasz Score: {calinski_avg:.3f}")
    
    return df, {
        'silhouette_score': silhouette_avg,
        'calinski_score': calinski_avg,
        'n_clusters': n_clusters,
        'cluster_stats': cluster_stats,
        'pca_components': pca.n_components_,
        'explained_variance_ratio': pca.explained_variance_ratio_
    }

def predictive_modeling(df):
    """Build and evaluate predictive models"""
    # Prepare features for prediction
    features = ['latitude', 'longitude', 'depth', 'day_of_year', 'month', 'year', 'hour']
    X = df[features]
    y = df['magnitude']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return {
        'rmse': rmse,
        'r2': r2,
        'feature_importance': feature_importance
    }

def save_results(df, stats, time_series, wavelet_results, clustering_metrics, model_metrics, output_dir='output'):
    """Save all results to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save processed data
    df.to_csv(f'{output_dir}/processed_data.csv', index=False)
    
    # Save statistics
    with open(f'{output_dir}/statistics.txt', 'w') as f:
        f.write("Earthquake Data Statistics\n")
        f.write("========================\n\n")
        f.write(f"Total Earthquakes: {stats['total_earthquakes']}\n")
        f.write(f"Date Range: {stats['date_range'][0]} to {stats['date_range'][1]}\n\n")
        f.write("Magnitude Statistics:\n")
        f.write(str(stats['magnitude_stats']))
        f.write("\n\nDepth Statistics:\n")
        f.write(str(stats['depth_stats']))
        
        # Add model metrics
        f.write("\n\nModel Performance Metrics:\n")
        f.write(f"RMSE: {model_metrics['rmse']:.4f}\n")
        f.write(f"R² Score: {model_metrics['r2']:.4f}\n")
        
        f.write("\nFeature Importance:\n")
        f.write(str(model_metrics['feature_importance']))
        
        # Add clustering metrics
        f.write("\n\nClustering Metrics:\n")
        f.write(f"Number of clusters: {clustering_metrics['n_clusters']}\n")
        f.write(f"Average Silhouette Score: {clustering_metrics['silhouette_score']:.3f}\n")
        f.write(f"Calinski-Harabasz Score: {clustering_metrics['calinski_score']:.3f}\n")
        f.write("\nCluster Statistics:\n")
        f.write(str(clustering_metrics['cluster_stats']))
    
    # Save time series data
    time_series['monthly_counts'].to_csv(f'{output_dir}/monthly_counts.csv')
    time_series['monthly_avg_magnitude'].to_csv(f'{output_dir}/monthly_avg_magnitude.csv')
    
    # Create and save visualizations
    plt.figure(figsize=(15, 10))
    
    # Time series plots
    plt.subplot(2, 2, 1)
    time_series['monthly_counts'].plot()
    plt.title('Monthly Earthquake Counts')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    time_series['monthly_avg_magnitude'].plot()
    plt.title('Monthly Average Magnitude')
    plt.grid(True)
    
    # Wavelet analysis plots
    plt.subplot(2, 2, 3)
    plt.plot(wavelet_results['daily_counts'].values)
    plt.title('Daily Earthquake Counts')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(wavelet_results['reconstructed_signals'][1])
    plt.title('Wavelet Decomposition (Level 1)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/analysis_plots.png')
    plt.close()
    
    # Clustering metrics plot
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, len(clustering_metrics['explained_variance_ratio']) + 1), clustering_metrics['explained_variance_ratio'])
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.grid(True)
    plt.savefig(f'{output_dir}/clustering_metrics.png')
    plt.close()

def main():
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Fetch and process data
    print("Fetching data from EMSC...")
    data = fetch_emsc_data()
    
    print("Preprocessing data...")
    df = preprocess_data(data)
    
    print("Generating descriptive statistics...")
    stats = descriptive_statistics(df)
    
    print("Performing time series analysis...")
    time_series = time_series_analysis(df)
    
    print("Performing wavelet analysis...")
    wavelet_results = wavelet_analysis(df)
    
    print("Performing clustering analysis...")
    df, clustering_metrics = clustering_analysis(df)
    
    print("Building predictive models...")
    model_metrics = predictive_modeling(df)
    
    print("Saving results...")
    save_results(df, stats, time_series, wavelet_results, clustering_metrics, model_metrics)
    
    print("Analysis complete! Results saved in the 'output' directory.")

if __name__ == "__main__":
    main()
