import pandas as pd

file_path = 'normalized_output.csv'
df = pd.read_csv(file_path)

# List of base columns to check against
columns_to_check = ['VolHull', 'AreaHull']
# List of suffixes to check
suffixes = ['_hull2', '_CGAL']

# Dictionary to hold filenames and implementations with significant differences for each base column
files_with_diff = {col: [] for col in columns_to_check}

for base in columns_to_check:
    for suffix in suffixes:
        col_name = f"{base}{suffix}"
        if col_name in df.columns:
            # Check for 1% difference
            difference = (df[col_name] - 1.0).abs() > 0.01
            if difference.any():
                for index in df[difference].index:
                    filename = df.loc[index, 'Filename']
                    files_with_diff[base].append((filename, col_name))
                    print(filename)
                    print(f"{base} : {df.loc[index,base]}")
                    print(f"{col_name} : {df.loc[index,col_name]}")

# Saving output to separate files based on the base being compared
for base in columns_to_check:
    output_filename = f'files_with_differences_{base}.txt'
    with open(output_filename, 'w') as file:
        for filename, implementation in files_with_diff[base]:
            file.write(f"{filename},{implementation}\n")
    print(f"Filenames with more than 1% difference in {base} saved to '{output_filename}'.")

print("Check complete.")
