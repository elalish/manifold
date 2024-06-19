import pandas as pd

file_path = 'normalized_output.csv'
df = pd.read_csv(file_path)

# Columns for statistics calculation
columns = ['VolHull', 'AreaHull', 'HullTri', 'Time']
# Columns suffixes to use
suffixes = ['', '_hull2', '_hull3', '_hull4', '_CGAL']

# Function to calculate statistics
def calculate_stats(column):
    mean_val = df[column].mean()
    median_val = df[column].median()
    mode_val = df[column].mode()
    max_val = df[column].max()
    min_val = df[column].min()
    
    if mode_val.empty:
        mode_val = None
    else:
        mode_val = mode_val.iloc[0]
    
    return mean_val, median_val, mode_val, max_val, min_val

stats_dict = {}

# Calculating stats for each column and their suffixes
for base in columns:
    for suffix in suffixes:
        col_name = f"{base}{suffix}"
        if col_name in df.columns:
            mean_val, median_val, mode_val, max_val, min_val = calculate_stats(col_name)
            stats_dict[col_name] = {
                'mean': mean_val,
                'median': median_val,
                'mode': mode_val,
                'max': max_val,
                'min': min_val
            }

# Converting the stats dictionary to a df for better visualization
stats_df = pd.DataFrame(stats_dict).T

stats_df.to_csv('statistics_output.csv')

print("Statistics calculation complete. Output saved to 'statistics_output.csv'.")
print(stats_df)
