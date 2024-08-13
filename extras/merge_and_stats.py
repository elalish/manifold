import pandas as pd


# MERGING THE DATA


filenames=[]
# merged_data = {} 
def parse_csv_and_merge(csv_files, output_file='merged_data.csv'):
    """
    Merges CSV files, handling multiline entries and various error conditions.

    Args:
        csv_files (list): List of tuples containing (filename, implementation_name).
        output_file (str, optional): Name of the output CSV file. Defaults to 'merged_data.csv'.
    """

    # merged_data = pd.DataFrame(columns=['Filename'])
    merged_data={}
    is_multiline = False
    multiline_data = []
    curr_file=""
    for file, implementation in csv_files:
        print(f"Starting File : {file}")
        try:
            df = pd.read_csv(file)
        except FileNotFoundError:
            print(f"Error: File '{file}' not found. Skipping...")
            continue

        for i, row in df.iterrows():
            if is_multiline:
                # Handling multiline entries (Before standard algorithm call)
                if 'After standard algorithm call' in row.values[0]:
                    is_multiline = True
                    continue
                elif row.values[1] == "Error":
                    row.fillna(0, inplace=True)  
                    row['Status'] = 'Error'
                    row.values[0]= curr_file
                    row.values[1] = 0
                    is_multiline=False
                    filename = row['Filename']
                    if filename not in merged_data:
                        merged_data[filename] = row.to_dict()  
                    else:
                        for col in df.columns: 
                          if col != 'Filename' and not pd.isna(row[col]):
                              merged_data[filename][col+"_"+implementation] = row[col]
                elif row.values[0] == "Invalid Output by algorithm":
                    is_multiline = True
                    continue
                else:
                    is_multiline = False
                    prev_item=curr_file
                    filenames.append(curr_file)
                    temp_item=row.values[0]
                    temp_len=row.values.size
                    for i in range(1,temp_len):
                      # print(temp_item)
                      temp_item=row.values[i-1]
                      row.values[i-1]=prev_item
                      prev_item=temp_item
                    # print(row)
                    filename = row['Filename']
                    if filename not in merged_data:
                        merged_data[filename] = row.to_dict()  
                    else:
                        for col in df.columns:  
                          if col != 'Filename' and not pd.isna(row[col]):
                              merged_data[filename][col+"_"+implementation] = row[col]
            else:
                # Handling single-line entries or first line of multiline entries
                # Checking for timeout or error
                if pd.isna(row['VolManifold']): 
                    if (row['VolHull']=="Timeout"):
                    # if 'Timeout' in row['Status']:
                        row['VolHull']=0
                        row['VolManifold'] = 0
                        row.fillna(0, inplace=True)  
                        row['Status'] = 'Timeout'
                    elif 'Error' in row['Status']:
                        row.fillna(0, inplace=True)
                        row['Status'] = 'Error'
                    elif (row['VolHull'] == "Error"):
                        row.fillna(0, inplace=True)
                        row['Status'] = 'Error'
                        pass
                    filename = row['Filename']
                    if filename not in merged_data:
                        merged_data[filename] = row.to_dict()
                    else:
                      for col in df.columns: 
                          if col != 'Filename' and not pd.isna(row[col]):
                              merged_data[filename][col+"_"+implementation] = row[col]
                    continue
                # Converting Series to df for renaming columns
                if 'Before standard algorithm call' in row.values[1]:
                    if row.values[2] == "Timeout":
                        row.fillna(0, inplace=True)
                        row['Status'] = 'Timeout'
                        row['VolHull']=0
                        row['VolManifold'] = 0
                        filename = row['Filename']
                        if filename not in merged_data:
                            merged_data[filename] = row.to_dict() 
                        else:
                            for col in df.columns:
                              if col != 'Filename' and not pd.isna(row[col]):
                                  merged_data[filename][col+"_"+implementation] = row[col]
                        continue
                    is_multiline = True
                    curr_file=row.values[0]
                else:
                  if (row['VolManifold']=="timeout: the monitored command dumped core"):
                        row.fillna(0, inplace=True)  
                        row['VolManifold']=0
                        row['VolHull'] = 0
                        row['Status'] = 'Error'
                  filename = row['Filename']
                  if filename not in merged_data:
                      merged_data[filename] = row.to_dict() 
                  else:
                      # print(merged_data[filename])
                      for col in df.columns:
                          if col != 'Filename' and not pd.isna(row[col]):
                              merged_data[filename][col+"_"+implementation] = row[col]

                    # multiline_data.append(row.tolist())
            # print(merged_data)

    if not merged_data:
        print("Warning: No valid data found in any CSV files.")
        return

    # Creating df from the dictionary to store the merged data
    merged_data = pd.DataFrame.from_dict(merged_data, orient='index')

    merged_data.to_csv(output_file, index=False)

csv_files = [('Hull1.csv','hull1'),('CGAL.csv', 'CGAL')]
parse_csv_and_merge(csv_files)


# NORMALIZE THE DATA


file_path = 'merged_data.csv'
df = pd.read_csv(file_path)

time_columns = [col for col in df.columns if 'Time' in col]
for col in time_columns:
    df[col] = df[col].str.replace(' sec', '').astype(float)

# List of base columns to normalize against
base_columns = ['VolManifold', 'VolHull', 'AreaManifold', 'AreaHull', 'ManifoldTri', 'HullTri', 'Time']
# List of suffixes to normalize
suffixes = ['_CGAL']
# for suffix in suffixes : 
# For time metric avoiding cases with time less than 0.001 seconds
# df = df[(df['Time'] > 0.001)]
# Normalize the columns and check for zero base values
stl_files_with_diff = []

for base in base_columns:
    base_col = base
    if base_col in df.columns:
        for suffix in suffixes:
            col_name = f"{base}{suffix}"
            if col_name in df.columns:
                # Checking if base column is zero and suffix column is not zero
                zero_base_nonzero_suffix = (df[base_col] == 0) & (df[col_name] != 0)
                if zero_base_nonzero_suffix.any():
                    raise ValueError(f"Error: {base_col} is zero while {col_name} is not zero in row(s): {df[zero_base_nonzero_suffix].index.tolist()}")
                
                # Setting col_name column in df to 1 if both are zero
                both_zero = (df[base_col] == 0) & (df[col_name] == 0)
                df.loc[both_zero, col_name] = 1
                
                # Normalizing the column while handling division by zero
                df[col_name] = df[col_name] / df[base_col].replace({0: 1})  

        df[base_col] = 1.0         


df.to_csv('normalized_output.csv', index=False)


# CALCULATE STATISTICS ON NORMALZIED OUTPUT


import pandas as pd

file_path = 'normalized_output.csv'
df = pd.read_csv(file_path)

# Columns for statistics calculation
columns = ['VolHull', 'AreaHull', 'HullTri', 'Time']
# Columns suffixes to use
suffixes = ['', '_CGAL']

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
