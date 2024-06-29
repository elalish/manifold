import pandas as pd

file_path = 'normalized_output.csv'
df = pd.read_csv(file_path)

# Columns for statistics calculation
columns = ['VolHull', 'AreaHull', 'HullTri', 'Time']
# Columns suffixes to use
suffixes = ['', '_hull2', '_hull3', '_hull4','_hull5', '_CGAL']

# Function to calculate statistics for each base and implementation
def calculate_stats(column, status,suffix):
    filtered_df = df[(df['Status'+suffix] == status) & ~df[column].isnull()]
    # filtered_df = df[(df['Status'+suffix] == status) & ~df[column].isnull() & (df['Time'+suffix] > 0.001) & (df['Time'] > 0.001)]
    success_count = filtered_df.shape[0]
    
    if success_count > 0:
        mean_val = filtered_df[column].mean()
        median_val = filtered_df[column].median()
        mode_val = filtered_df[column].mode().iloc[0] if not filtered_df[column].mode().empty else None
        max_val = filtered_df[column].max()
        min_val = filtered_df[column].min()
    else:
        mean_val = median_val = mode_val = max_val = min_val = None

    return mean_val, median_val, mode_val, max_val, min_val, success_count

stats_dict = {}

# Calculating stats for each column and their suffixes
for base in columns:
    for suffix in suffixes:
        col_name = f"{base}{suffix}"
        if col_name in df.columns:
            mean_val, median_val, mode_val, max_val, min_val, success_count = calculate_stats(col_name, 'Success',suffix)
            stats_dict[col_name] = {
                'mean': mean_val,
                'median': median_val,
                'mode': mode_val,
                'max': max_val,
                'min': min_val,
                'Success_Count': success_count
            }

# Converting the stats dictionary to a df for better visualization
stats_df = pd.DataFrame(stats_dict).T

stats_df.to_csv('statistics_output.csv')

print("Statistics calculation complete. Output saved to 'statistics_output.csv'.")
print(stats_df)
