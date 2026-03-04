import pandas as pd
import numpy as np
def analyze_baidu_index(csv_path, output_dir='./'):
    """
    Threshold screening to eliminate low-value cities
    
    Parameters:
    csv_path: Input CSV file path
    output_dir: Output file save directory
    
    Returns:
    filtered_cities: Statistical information of qualified cities
    filtered_monthly_data: Monthly data of qualified cities
    """
    
    # 1. Read data
    print("Reading data...")
    df = pd.read_csv(csv_path)
    print(f"Data reading completed, total {len(df):,} records, {len(df.columns)} columns")
    print(f"Data column names: {df.columns.tolist()}")
    print(f"Data time range: {df['Year-Month'].min()} to {df['Year-Month'].max()}")
    
    # 2. Calculate statistical indicators by city grouping
    print("\nCalculating statistical indicators for each city...")
    city_stats = df.groupby('City')['Value'].agg([
        'mean',    # Monthly average
        'max',     # Maximum value
        'count',   # Number of data months
        'std'      # Standard deviation (additional statistics)
    ]).reset_index()
    
    # Rename columns
    city_stats.columns = ['City', 'Monthly_Avg', 'Max_Value', 'Data_Months', 'Std_Dev']
    
    print(f"Statistical analysis completed for {len(city_stats)} cities' Baidu Index data")
    
    # 3. Filter qualified cities
    print("\nFiltering qualified cities...")
    # Filter criteria: Monthly average ≥ 3200 or Maximum value ≥ 9000
    filtered_cities = city_stats[
        (city_stats['Monthly_Avg'] >= 3200) | 
        (city_stats['Max_Value'] >= 9000)
    ].sort_values('Monthly_Avg', ascending=False)
    
    print(f"Filtering completed, {len(filtered_cities)} cities meet the criteria in total")
    
    # 4. Obtain complete monthly data of filtered cities
    print("\nExtracting monthly data of qualified cities...")
    filtered_city_names = filtered_cities['City'].tolist()
    filtered_monthly_data = df[df['City'].isin(filtered_city_names)].copy()
    
    # Sort by city and time
    filtered_monthly_data['Year-Month_Sort'] = pd.to_datetime(filtered_monthly_data['Year-Month'] + '-01')
    filtered_monthly_data = filtered_monthly_data.sort_values(['City', 'Year-Month_Sort'])
    filtered_monthly_data = filtered_monthly_data.drop('Year-Month_Sort', axis=1)
    
    print(f"Extracted {len(filtered_monthly_data):,} monthly data records in total")
    
    # 5. Save results to files
    print("\nSaving result files...")
    # Save statistical information
    stats_file = f"{output_dir}/Filtered_Cities_Statistics.csv"
    filtered_cities.to_csv(stats_file, index=False, encoding='utf-8-sig')
    
    # Save monthly data
    monthly_file = f"{output_dir}/Filtered_Cities_Monthly_Data.csv"
    filtered_monthly_data.to_csv(monthly_file, index=False, encoding='utf-8-sig')
    
    print(f"Result files saved:")
    print(f"1. Statistical information: {stats_file}")
    print(f"2. Monthly data: {monthly_file}")
    
    # 6. Output analysis summary
    print(f"\n{'='*60}")
    print("Analysis Result Summary")
    print(f"{'='*60}")
    
    # Conditional distribution statistics
    cond1_only = filtered_cities[(filtered_cities['Monthly_Avg'] >= 3200) & (filtered_cities['Max_Value'] < 9000)]
    cond2_only = filtered_cities[(filtered_cities['Monthly_Avg'] < 3200) & (filtered_cities['Max_Value'] >= 9000)]
    both_cond = filtered_cities[(filtered_cities['Monthly_Avg'] >= 3200) & (filtered_cities['Max_Value'] >= 9000)]
    
    print(f"1. Filter criteria distribution:")
    print(f"   - Monthly average ≥ 3200 only: {len(cond1_only)} cities")
    print(f"   - Maximum value ≥ 9000 only: {len(cond2_only)} cities")
    print(f"   - Meet both criteria: {len(both_cond)} cities")
    print(f"   - Total: {len(filtered_cities)} cities")
    
    print(f"\n2. Top 10 cities by monthly average:")
    top10 = filtered_cities.head(10)
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        print(f"   {i:2d}. {row['City']:>4s} - Monthly Avg: {row['Monthly_Avg']:>8.2f}, Max Value: {row['Max_Value']:>9.2f}")
    
    return filtered_cities, filtered_monthly_data
# Main program execution
if __name__ == "__main__":
    # Input file path
    input_file = "./00Baidu_Index_Monthly_Avg (Full_Sample).csv"
    
    # Execute analysis
    result_stats, result_monthly = analyze_baidu_index(input_file)
    
    print(f"\nAnalysis completed!")