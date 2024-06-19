import pandas as pd

df = pd.read_csv('merged_data.csv')

algorithm_counts = {
    '' : 0,
    '_hull3': 0,
    '_hull2': 0,
    '_hull4': 0,
    '_CGAL': 0
}

# Iterating through each row in the df
for index, row in df.iterrows():
    filename = row['Filename']
    # Initializing the fastest algorithm and time taken
    fastest_algorithm = None
    fastest_time = float('inf') 
    # Iterating through columns corresponding to each algorithm
    for algorithm in ['','_hull3', '_hull2', '_hull4', '_CGAL']:
        time_column = f'Time{algorithm}'
        # print(row)
        if type(row[time_column]) is str:
          time_value = float(row[time_column].split()[0])
        else:
          time_value=float(row[time_column])
        # Checking if the time value is not empty and less than the current fastest time
        if time_value and time_value < fastest_time:
            fastest_algorithm = algorithm
            fastest_time = time_value
    if fastest_algorithm:
        algorithm_counts[fastest_algorithm] += 1

for algorithm, count in algorithm_counts.items():
    print(f'Algorithm {algorithm}: {count} times fastest')

