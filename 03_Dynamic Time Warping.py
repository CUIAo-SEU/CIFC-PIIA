import pandas as pd
import numpy as np
from tslearn.clustering import TimeSeriesKMeans
import matplotlib.pyplot as plt
def cluster_time_series_by_dtw(csv_path, output_dir='./', num_clusters=2):
    """
    Time series K-Means clustering analysis of city monthly data based on DTW (Dynamic Time Warping)
    
    Parameters:
    csv_path: str - Input CSV file path
    output_dir: str - Output directory
    num_clusters: int - Number of clusters
    
    Returns:
    cluster_result_df: pd.DataFrame - Clustering results of each city (with serial number)
    raw_data: pd.DataFrame - Raw data
    time_series_list: list - List of time series array grouped by city
    kmeans_model: TimeSeriesKMeans - Trained clustering model
    """
    # 1. Read CSV file
    print("Reading data...")
    raw_data = pd.read_csv(csv_path)
    print(f"Data reading completed, total {len(raw_data):,} records, {len(raw_data.columns)} columns")
    print(f"Data column names: {raw_data.columns.tolist()}")
    if 'Year-Month' in raw_data.columns:
        print(f"Data time range: {raw_data['Year-Month'].min()} to {raw_data['Year-Month'].max()}")
    
    # 2. Data preprocessing: convert to time series format by city grouping
    print("\nPreprocessing time series data...")
    # Extract value series by city grouping
    city_groups = raw_data.groupby('City')['Value'].apply(list)
    time_series_list = [np.array(ts).reshape(-1, 1) for ts in city_groups.tolist()]
    cities = city_groups.index.tolist()
    
    print(f"Time series preprocessing completed: {len(cities)} cities in total, sequence length of each city: {len(time_series_list[0]) if time_series_list else 0}")
    
    # 3. Execute K-Means clustering with DTW distance
    print(f"\nExecuting DTW-KMeans clustering (number of clusters: {num_clusters})...")
    kmeans_model = TimeSeriesKMeans(
        n_clusters=num_clusters,
        metric="dtw",
        random_state=42,
        n_jobs=-1  # Enable multi-threading acceleration
    )
    cluster_labels = kmeans_model.fit_predict(time_series_list)
    print("DTW-KMeans clustering training completed")
    
    # 4. Organize clustering results into structured DataFrame
    print("\nOrganizing clustering results...")
    cluster_result_df = pd.DataFrame({
        'City': cities,
        'Cluster_Label': cluster_labels
    }).sort_values(by=['Cluster_Label', 'City']).reset_index(drop=True)
    
    # Add serial number column (increment from 1)
    cluster_result_df.insert(0, 'Serial_Number', range(1, len(cluster_result_df) + 1))
    
    # Count the number of cities in each cluster
    cluster_count = cluster_result_df['Cluster_Label'].value_counts().sort_index()
    print("Clustering result organization completed")
    
    # -------------------------- Key statistical information output --------------------------
    print(f"\n{'='*60}")
    print("DTW-KMeans Clustering Key Statistical Information Summary")
    print(f"{'='*60}")
    print(f"Number of clusters: {num_clusters}")
    for cluster_id in range(num_clusters):
        city_num = cluster_count.get(cluster_id, 0)
        city_list = cluster_result_df[cluster_result_df['Cluster_Label'] == cluster_id]['City'].tolist()
        print(f"Cluster {cluster_id}: {city_num} cities | City list: {', '.join(city_list)}")
    print(f"Total clustered cities: {len(cities)}")
    
    # Output complete clustering results of all cities
    print("\n=== Complete List of Clustering Results for All Cities ===")
    print(cluster_result_df.to_string(index=False))
    
    return cluster_result_df, raw_data, time_series_list, kmeans_model
# Main program execution
if __name__ == "__main__":
    # Input file path
    input_file = "02Baidu_Index_Monthly_Avg (Remaining_59_Internet-Famous_Cities).csv"
    
    # Execute DTW time series clustering analysis
    print("Starting DTW-based time series K-Means clustering analysis for cities...")
    print(f"Input file path: {input_file}")
    print("-" * 80)
    
    # Call clustering function
    cluster_result, raw_data, ts_list, kmeans_model = cluster_time_series_by_dtw(
        csv_path=input_file,
        num_clusters=2
    )
    
    print("-" * 80)
    print("\nClustering analysis completed!")