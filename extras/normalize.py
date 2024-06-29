import pandas as pd

file_path = 'merged_data.csv'
df = pd.read_csv(file_path)

time_columns = [col for col in df.columns if 'Time' in col]
for col in time_columns:
    df[col] = df[col].str.replace(' sec', '').astype(float)

# List of base columns to normalize against
base_columns = ['VolManifold', 'VolHull', 'AreaManifold', 'AreaHull', 'ManifoldTri', 'HullTri', 'Time']
# List of suffixes to normalize
suffixes = ['_hull2', '_hull3', '_hull4', '_hull5','_CGAL']
# for suffix in suffixes : 
# For time metric avoiding cases with time less than 0.001 seconds
df = df[(df['Time'] > 0.001)]
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


