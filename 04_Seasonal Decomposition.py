import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import re
warnings.filterwarnings('ignore')
def calc_city_seasonal_std(csv_path, output_dir='./', period=12, model='additive'):
    """
    Calculate the standard deviation of seasonal components of Baidu Index for each city
    based on time series seasonal decomposition
    
    Parameters:
    csv_path: str - Input CSV file path
    output_dir: str - Output directory for results and charts
    period: int - Seasonal period
    model: str - Seasonal decomposition model
    
    Returns:
    seasonal_std_df: pd.DataFrame - Standard deviation results of seasonal components for each city
    raw_data: pd.DataFrame - Preprocessed raw data
    decompose_info: dict - Key information of decomposition process
    """
    # 1. Read and check raw data
    print("Reading raw data...")
    raw_data = pd.read_csv(csv_path)
    # Basic information output
    print(f"Data reading completed: total {len(raw_data):,} records, {len(raw_data.columns)} columns")
    print(f"Data column names: {raw_data.columns.tolist()}")
    if 'Year-Month' in raw_data.columns and 'City' in raw_data.columns:
        time_range = f"{raw_data['Year-Month'].min()} to {raw_data['Year-Month'].max()}"
        city_count = raw_data['City'].nunique()
        print(f"Data time range: {time_range}")
        print(f"Number of involved cities: {city_count}")
    
    # 2. Data preprocessing
    print("\nPerforming data preprocessing...")
    # Delete missing values in 'Year-Month', 'City' and 'Value' columns (avoid invalid data interference)
    processed_data = raw_data.dropna(subset=['Year-Month', 'City', 'Value']).copy()
    
    # Fix: convert time format to datetime (process multiple date formats)
    def parse_date(date_str):
        """
        Parse multiple possible date formats
        """
        if pd.isna(date_str):
            return pd.NaT
        
        date_str = str(date_str).strip()
        
        # Try common formats
        patterns = [
            (r'([A-Za-z]+)-(\d{2,4})', '%b-%y'),  # Jan-11, Jan-2011
            (r'(\d{4})-(\d{1,2})', '%Y-%m'),      # 2011-01
            (r'(\d{4})/(\d{1,2})', '%Y/%m'),      # 2011/01
            (r'(\d{4})年(\d{1,2})月', '%Y年%m月'), # 2011年1月
            (r'(\d{1,2})-(\d{4})', '%m-%Y'),      # 01-2011
            (r'(\d{1,2})/(\d{4})', '%m/%Y'),      # 01/2011
        ]
        
        for pattern, fmt in patterns:
            match = re.search(pattern, date_str)
            if match:
                try:
                    # For Jan-11 format
                    if fmt == '%b-%y':
                        month_str, year_str = match.groups()
                        # Process two or four digit year
                        if len(year_str) == 2:
                            # Assume 11 represents 2011 (21st century)
                            year_int = int(year_str)
                            if year_int < 50:  # 50 as the dividing line
                                year_str = '20' + year_str
                            else:
                                year_str = '19' + year_str
                        # Try parsing
                        return pd.to_datetime(f"{month_str} {year_str}", format='%b %Y')
                    # For other formats
                    else:
                        clean_str = match.group(0)
                        return pd.to_datetime(clean_str, format=fmt)
                except Exception as e:
                    continue
        
        # If none of the above match, try automatic parsing
        try:
            return pd.to_datetime(date_str, errors='coerce')
        except:
            return pd.NaT
    
    # Apply date parsing function
    processed_data['Year-Month'] = processed_data['Year-Month'].apply(parse_date)
    
    # Check parsing results
    date_na_count = processed_data['Year-Month'].isna().sum()
    if date_na_count > 0:
        print(f"Warning: {date_na_count} dates failed to parse, set to NaT")
        # Display first few failed parsing dates
        failed_dates = raw_data.loc[processed_data['Year-Month'].isna(), 'Year-Month'].head().tolist()
        print(f"Examples of failed parsing dates: {failed_dates}")
        # Delete rows with invalid dates
        processed_data = processed_data.dropna(subset=['Year-Month'])
        print(f"Deleted {date_na_count} records with invalid dates")
    
    # Ensure time column is datetime type and sort by city + time
    processed_data['Year-Month'] = pd.to_datetime(processed_data['Year-Month'])
    processed_data = processed_data.sort_values(by=['City', 'Year-Month']).reset_index(drop=True)
    
    # Check data time range
    print(f"Processed time range: {processed_data['Year-Month'].min()} to {processed_data['Year-Month'].max()}")
    print(f"Processed data volume: {len(processed_data):,} records")
    
    # 3. Data pivot: row=time, column=city, value=Baidu Index (standardized time series format)
    ts_pivot = processed_data.pivot(index='Year-Month', columns='City', values='Value')
    cities = ts_pivot.columns.tolist()  # List of all cities
    valid_time_len = len(ts_pivot)      # Number of valid time points
    print(f"\nData pivot completed: time series length {valid_time_len} months, {len(cities)} cities to be analyzed")
    
    # 4. Batch seasonal decomposition and standard deviation calculation (core calculation link)
    print(f"\nStarting seasonal decomposition (period: {period} months, model: {model})...")
    seasonal_std_dict = {}  # Store standard deviation results of each city
    success_count = 0       # Number of cities with successful decomposition
    fail_count = 0          # Number of cities with failed decomposition
    short_data_count = 0    # Number of cities with insufficient data
    fail_details = []       # Record failure details
    
    # Process each city one by one
    for i, city in enumerate(cities, 1):
        # Extract time series of a single city (delete sporadic missing values inside the city)
        city_ts = ts_pivot[city].dropna()
        
        # Data volume verification: at least 2 periods are required for effective decomposition
        if len(city_ts) < 2 * period:
            seasonal_std_dict[city] = "Insufficient Data"
            short_data_count += 1
            fail_details.append((city, f"Insufficient Data ({len(city_ts)} < {2*period})"))
            continue
        
        # Perform seasonal decomposition and calculate standard deviation
        try:
            # Use extrapolate_trend='freq' to handle boundary issues
            decomposition = seasonal_decompose(
                city_ts, 
                model=model, 
                period=period,
                extrapolate_trend='freq'
            )
            # Calculate standard deviation of seasonal components (retain 2 decimal places for readability)
            seasonal_std = round(decomposition.seasonal.dropna().std(), 2)
            seasonal_std_dict[city] = seasonal_std
            success_count += 1
            
            # Progress display
            if i % 10 == 0 or i == len(cities):
                print(f"  Processed {i}/{len(cities)} cities, {success_count} successful")
                
        except Exception as e:
            # Capture exceptions during decomposition
            seasonal_std_dict[city] = "Decomposition Failed"
            fail_count += 1
            fail_details.append((city, str(e)[:100]))  # Only record the first 100 characters
    
    # Organize decomposition process information
    decompose_info = {
        "Total_Cities": len(cities),
        "Successful_Decomposition": success_count,
        "Insufficient_Data": short_data_count,
        "Decomposition_Failed": fail_count,
        "Time_Series_Length": valid_time_len,
        "Seasonal_Period": period,
        "Decomposition_Model": model,
        "Failure_Details": fail_details[:10]  # Only retain the first 10 failure details
    }
    
    print(f"\nDecomposition completed: {success_count} successful | {short_data_count} insufficient data | {fail_count} failed")
    
    if fail_details:
        print("First 5 failure details:")
        for city, reason in fail_details[:5]:
            print(f"  {city}: {reason}")
    
    # 5. Result structuring and sorting
    print("\nOrganizing and sorting results...")
    # Convert to DataFrame format
    seasonal_std_df = pd.DataFrame({
        "City": list(seasonal_std_dict.keys()),
        "Seasonal_Component_Std": list(seasonal_std_dict.values())
    })
    
    # Separate numeric results and text results
    numeric_mask = pd.to_numeric(seasonal_std_df["Seasonal_Component_Std"], errors="coerce").notna()
    numeric_results = seasonal_std_df[numeric_mask].copy()
    non_numeric_results = seasonal_std_df[~numeric_mask].copy()
    
    # Sort numeric results in descending order of standard deviation, text results in alphabetical order of city name
    if not numeric_results.empty:
        numeric_results["Seasonal_Component_Std"] = numeric_results["Seasonal_Component_Std"].astype(float)
        numeric_results_sorted = numeric_results.sort_values(by="Seasonal_Component_Std", ascending=False)
    else:
        numeric_results_sorted = pd.DataFrame()
    
    non_numeric_results_sorted = non_numeric_results.sort_values(by="City")
    
    # Merge sorted results
    seasonal_std_df = pd.concat([numeric_results_sorted, non_numeric_results_sorted]).reset_index(drop=True)
    # Add serial number column
    seasonal_std_df.insert(0, "Serial_Number", range(1, len(seasonal_std_df) + 1))
    
    # 6. Key result statistics and output
    print(f"\n{'='*60}")
    print("City Seasonal Component Standard Deviation Calculation Result Statistics")
    print(f"{'='*60}")
    
    # Only count key indicators of valid numeric results
    if not numeric_results.empty:
        max_std_city = numeric_results_sorted.iloc[0]["City"]
        max_std_value = numeric_results_sorted.iloc[0]["Seasonal_Component_Std"]
        min_std_city = numeric_results_sorted.iloc[-1]["City"]
        min_std_value = numeric_results_sorted.iloc[-1]["Seasonal_Component_Std"]
        avg_std_value = round(numeric_results_sorted["Seasonal_Component_Std"].mean(), 2)
        median_std_value = round(numeric_results_sorted["Seasonal_Component_Std"].median(), 2)
        
        print(f"Maximum standard deviation: {max_std_value} ({max_std_city})")
        print(f"Minimum standard deviation: {min_std_value} ({min_std_city})")
        print(f"Average standard deviation: {avg_std_value}")
        print(f"Median standard deviation: {median_std_value}")
        print(f"Number of cities with valid statistics: {len(numeric_results)}")
        
        # Display distribution
        print(f"\nSeasonal component standard deviation distribution:")
        quantiles = [0.25, 0.5, 0.75, 0.9]
        for q in quantiles:
            value = round(numeric_results_sorted["Seasonal_Component_Std"].quantile(q), 2)
            print(f"  {int(q*100)}th percentile: {value}")
    else:
        print("Warning: No valid numeric results")
    
    # Output complete results of all cities
    print(f"\n{'='*60}")
    print("List of All City Results (First 20 Rows)")
    print(f"{'='*60}")
    print(seasonal_std_df.head(20).to_string(index=False))
    
    if len(seasonal_std_df) > 20:
        print(f"... and {len(seasonal_std_df) - 20} other cities")
    
    # 7. Save results (CSV file)
    csv_output_path = f"{output_dir}/City_Seasonal_Component_Std_Results.csv"
    seasonal_std_df.to_csv(csv_output_path, index=False, encoding="utf-8-sig")
    print(f"\nResult file saved to: {csv_output_path}")
    
    
    return seasonal_std_df, processed_data, decompose_info
# Main program execution
if __name__ == "__main__":
    # Input and output configuration (adjust according to actual path)
    input_file = "03Baidu_Index_Monthly_Avg (Final_Remaining_46_Internet-Famous_Cities).csv"  # Your input file path
    output_directory = "./"                 # Result output directory
    
    print("Starting city seasonal component standard deviation calculation...")
    print(f"Input file path: {input_file}")
    print("-" * 80)
    
    try:
        # Call core function
        result_df, processed_data, decompose_info = calc_city_seasonal_std(
            csv_path=input_file,
            output_dir=output_directory,
            period=12,
            model="additive"
        )
        
        print("-" * 80)
        print("\nCalculation completed! All results saved to the specified directory.")
        # Additional output of key process information
        print(f"Process summary: {decompose_info['Successful_Decomposition']}/{decompose_info['Total_Cities']} cities calculated successfully")
        
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        print("Please check if the file path is correct.")
    except Exception as e:
        print(f"Error occurred during execution: {e}")
        import traceback
        traceback.print_exc()