import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import re
def analyze_peak_values(csv_path, output_dir='./'):
    """
    Analyze moving average peaks of monthly data for potential 92 internet-famous cities,
    count the number of valid peaks for each city
    
    Parameters:
    csv_path: str - Input CSV file path
    
    Returns:
    result_df: pd.DataFrame - Statistical results of valid peak numbers for each city (with serial number)
    processed_data: pd.DataFrame - Complete processed data (including moving average)
    peak_results: dict - Original peak values of each city
    top_20_threshold: float - Top 20% threshold of all peaks (80th percentile)
    """
    # 1. Read CSV file
    print("Reading data...")
    red_cities_data = pd.read_csv(csv_path)
    print(f"Data reading completed, total {len(red_cities_data):,} records, {len(red_cities_data.columns)} columns")
    print(f"Data column names: {red_cities_data.columns.tolist()}")
    
    
    # 2. Preprocess date column
    
    def parse_date(date_str):
        """
        Parse multiple possible date formats
        """
        if pd.isna(date_str):
            return pd.NaT
        
        date_str = str(date_str).strip()
        
        # Try common formats
        patterns = [
            (r'(\d{4})-(\d{1,2})', '%Y-%m'),  # 2023-1, 2023-01
            (r'(\d{4})/(\d{1,2})', '%Y/%m'),  # 2023/1, 2023/01
            (r'(\d{4})年(\d{1,2})月', '%Y年%m月'),  # 2023年1月
            (r'(\d{1,2})-(\d{4})', '%m-%Y'),  # 1-2023, 01-2023
            (r'(\d{1,2})/(\d{4})', '%m/%Y'),  # 1/2023, 01/2023
            (r'([A-Za-z]+)-(\d{2,4})', '%b-%y'),  # Jan-23, Jan-2023
        ]
        
        for pattern, fmt in patterns:
            match = re.search(pattern, date_str)
            if match:
                try:
                    # For letter month format
                    if fmt == '%b-%y':
                        month_str, year_str = match.groups()
                        # Process two or four digit year
                        if len(year_str) == 2:
                            year_str = '20' + year_str  # Assume 20xx
                        return pd.to_datetime(f"{month_str}-{year_str}", format='%b-%Y')
                    # For other formats
                    else:
                        # Extract date part from original string
                        clean_str = match.group(0)
                        return pd.to_datetime(clean_str, format=fmt)
                except:
                    continue
    
        try:
            return pd.to_datetime(date_str, errors='coerce')
        except:
            return pd.NaT
    
    # Apply date parsing function
    red_cities_data['Year-Month'] = red_cities_data['Year-Month'].apply(parse_date)
    
    # Check parsing results
    date_na_count = red_cities_data['Year-Month'].isna().sum()
    print(f"Date conversion completed, {date_na_count} dates failed to parse, set to NaT")
    
    # Delete rows with invalid dates
    if date_na_count > 0:
        original_count = len(red_cities_data)
        red_cities_data = red_cities_data.dropna(subset=['Year-Month'])
        print(f"Deleted {original_count - len(red_cities_data)} records with invalid dates")
    
    # Extract year-month information as string (for display)
    red_cities_data['Year-Month_Str'] = red_cities_data['Year-Month'].dt.strftime('%Y-%m')
    
    print(f"Valid date time range: {red_cities_data['Year-Month'].min()} to {red_cities_data['Year-Month'].max()}")
    print(f"Processed data volume: {len(red_cities_data):,} records")
    
    # 3. Sort by city and ascending date
    print("\nSorting data by city and date...")
    red_cities_data = red_cities_data.sort_values(by=['City', 'Year-Month'])
    print("Data sorting completed")
    
    # 4. Process data with moving average method (time window 3)
    print("\nCalculating moving average (window size 3)...")
    red_cities_data['Moving_Avg'] = red_cities_data.groupby('City')['Value'].rolling(
        window=3, min_periods=1  # min_periods=1 ensures calculation with less than 3 data points
    ).mean().reset_index(level=0, drop=True)
    print(f"Moving average calculation completed, new column added: 'Moving_Avg'")
    
    # 5. Count peak numbers and peak values of moving average for each city
    print("\nIdentifying peaks of moving average for each city...")
    peak_results = {}
    peak_positions = {}  # Store peak positions
    
    for city, group in red_cities_data.groupby('City'):
        # Ensure data is sorted by time
        group = group.sort_values('Year-Month')
        
        # Identify peak positions
        peaks, properties = find_peaks(group['Moving_Avg'].values)
        
        # Extract values corresponding to peaks
        peak_values = group['Moving_Avg'].values[peaks]
        peak_results[city] = peak_values
        
        # Record peak positions (for debugging)
        peak_positions[city] = {
            'indices': peaks,
            'dates': group['Year-Month'].iloc[peaks].tolist() if len(peaks) > 0 else []
        }
    
    cities_with_peaks = sum(1 for v in peak_results.values() if len(v) > 0)
    total_peaks = sum(len(v) for v in peak_results.values())
    print(f"Peak identification completed, processed {len(peak_results)} cities in total, {cities_with_peaks} cities detected with peaks, total peaks: {total_peaks}")
    
    # 6. Calculate top 20% threshold of all peaks (80th percentile)
    print("\nCalculating top 20% threshold of peaks (80th percentile)...")
    if total_peaks > 0:
        all_peak_values = np.concatenate(list(peak_results.values()))
        top_20_threshold = np.percentile(all_peak_values, 80)  # 80th percentile is the critical value of top 20%
        print(f"Threshold calculation completed: {top_20_threshold:.2f} (total peaks: {len(all_peak_values)})")
    else:
        top_20_threshold = 0
        print("No peaks detected, threshold set to 0")
    
    # 7. Filter valid peaks and count the number
    print("\nFiltering valid peaks...")
    valid_peaks_count = {}
    valid_peaks_details = {}  # Store valid peak details
    
    for city, peak_values in peak_results.items():
        # Sort peaks in descending order
        sorted_peak_values = np.sort(peak_values)[::-1]
        valid_peaks = []
        
        if sorted_peak_values.size > 0:
            # The first peak is valid by default
            valid_peaks.append(sorted_peak_values[0])
            
            # Subsequent peaks need to meet either condition: ≥80% of the previous peak or ≥top 20% threshold
            for i in range(1, len(sorted_peak_values)):
                condition1 = sorted_peak_values[i] >= 0.8 * sorted_peak_values[i - 1]
                condition2 = sorted_peak_values[i] >= top_20_threshold
                
                if condition1 or condition2:
                    valid_peaks.append(sorted_peak_values[i])
                else:
                    break  # Subsequent peaks are smaller, terminate judgment
        
        valid_peaks_count[city] = len(valid_peaks)
        valid_peaks_details[city] = valid_peaks
    
    total_valid_peaks = sum(valid_peaks_count.values())
    print(f"Valid peak filtering completed, total valid peaks of all cities: {total_valid_peaks}")
    
    # 8. Organize results (add serial number column, retain all cities completely)
    print("\nOrganizing final results...")
    result_df = pd.DataFrame(
        list(valid_peaks_count.items()),
        columns=['City', 'Valid_Peak_Count']
    ).sort_values(by='Valid_Peak_Count', ascending=False).reset_index(drop=True)
    
    # Add "Serial_Number" column (increment from 1 for easy reference)
    result_df.insert(0, 'Serial_Number', range(1, len(result_df) + 1))
    
    # Calculate statistical information
    print("Result organization completed")
    
    # -------------------------- Key statistical information output --------------------------
    print(f"\n{'='*60}")
    print("Key Statistical Information Summary")
    print(f"{'='*60}")
    print(f"Top 20% threshold of all peaks: {top_20_threshold:.2f}")
    print(f"Total processed cities: {len(result_df)}")
    print(f"Total valid peaks: {total_valid_peaks}")
    print(f"Average valid peaks per city: {total_valid_peaks/len(result_df):.2f}")
    
    if len(result_df) > 0:
        print(f"City with the most valid peaks: {result_df.iloc[0]['City']} ({result_df.iloc[0]['Valid_Peak_Count']} peaks)")
        print(f"City with the fewest valid peaks: {result_df.iloc[-1]['City']} ({result_df.iloc[-1]['Valid_Peak_Count']} peaks)")
        
        # Display valid peak count distribution
        print(f"\nValid peak count distribution:")
        for count in sorted(result_df['Valid_Peak_Count'].unique()):
            cities_with_count = len(result_df[result_df['Valid_Peak_Count'] == count])
            print(f"  {count} peaks: {cities_with_count} cities")
    
    # Output complete valid peak count of all cities
    print(f"\n{'='*60}")
    print("Complete List of Valid Peak Counts for All Cities")
    print(f"{'='*60}")
    print(result_df.to_string(index=False))
    
    # Save results to CSV file
    output_file = f"{output_dir}/City_Valid_Peak_Count_Statistics.csv"
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nResults saved to: {output_file}")
    
    return result_df, red_cities_data, peak_results, top_20_threshold, valid_peaks_details
# Main program execution
if __name__ == "__main__":
    # Input file path
    input_file = "./01Baidu_Index_Monthly_Avg (Potential_92_Internet-Famous_Cities).csv"
    
    # Execute peak analysis
    print("Starting moving average peak analysis for cities...")
    print(f"Input file path: {input_file}")
    print("-" * 80)
    
    # Call analysis function
    try:
        result_df, processed_data, peak_results, threshold, valid_details = analyze_peak_values(input_file)
        
        print("-" * 80)
        print("\nAnalysis completed!")
        
        
    except Exception as e:
        print(f"Error occurred during execution: {e}")
        import traceback
        traceback.print_exc()