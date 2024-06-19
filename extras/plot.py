import pandas as pd
import matplotlib.pyplot as plt

file_path = 'normalized_output.csv'
df = pd.read_csv(file_path)

# Filtering out rows where VolHull_hull2 is 0 (Timeout cases)
df = df[df['VolHull_hull2'] != 0]

# Exclude rows where the absolute difference between any two columns is greater than 0.01% Used to look at the minute trends
df = df[((df['VolHull_CGAL'] - df['VolHull']).abs() < 0.0001) & ((df['VolHull_CGAL'] - df['VolHull_hull2']).abs() < 0.0001) & ((df['VolHull'] - df['VolHull_hull2']).abs() < 0.0001)]
print(df)

# Plotting
plt.plot(df.index, df['VolHull_CGAL'], label='VolHull_CGAL',marker='o')
plt.plot(df.index, df['VolHull_hull2'], label='VolHull_hull2',marker='o')
plt.plot(df.index, df['VolHull'], label='VolHull',marker='o')
plt.xlabel('Index')
plt.ylabel('Volume')
plt.title('Volume Comparison')
plt.legend()
plt.show()

#Scatter Plot

plt.scatter(df['AreaHull_CGAL'], df['VolHull_CGAL'], label='VolHull_CGAL', marker='o')
plt.scatter(df['AreaHull_hull2'], df['VolHull_hull2'], label='VolHull_hull2', marker='o')
plt.scatter(df['AreaHull'], df['VolHull'], label='VolHull', marker='o')

plt.xlabel('AreaHull')
plt.ylabel('Volume')
plt.title('Volume vs AreaHull Comparison')
plt.legend()
plt.show()


file_path = 'normalized_output.csv'
df = pd.read_csv(file_path)

# Filtering out rows where AreaHull_hull2 is 0 (Timeout cases)
df = df[df['VolHull_hull2'] != 0]

# Exclude rows where the absolute difference between any two columns is less than 0.01% Used to look at the outlier trends
df = df[((df['VolHull_CGAL'] - df['VolHull']).abs() > 0.0001) | ((df['VolHull_CGAL'] - df['VolHull_hull2']).abs() > 0.0001) | ((df['VolHull'] - df['VolHull_hull2']).abs() > 0.0001)]
print(df)

# Plotting
plt.plot(df.index, df['VolHull_CGAL'], label='VolHull_CGAL',marker='o')
plt.plot(df.index, df['VolHull_hull2'], label='VolHull_hull2',marker='o')
plt.plot(df.index, df['VolHull'], label='VolHull',marker='o')
plt.xlabel('Index')
plt.ylabel('Volume')
plt.title('Volume Comparison')
plt.legend()
plt.show()

#ScatterPlot
plt.scatter(df['AreaHull_CGAL'], df['VolHull_CGAL'], label='VolHull_CGAL', marker='o')
plt.scatter(df['AreaHull_hull2'], df['VolHull_hull2'], label='VolHull_hull2', marker='o')
plt.scatter(df['AreaHull'], df['VolHull'], label='VolHull', marker='o')

plt.xlabel('AreaHull')
plt.ylabel('Volume')
plt.title('Volume vs AreaHull Comparison')
plt.legend()
plt.show()

#Storing the output to the csv
df.to_csv('diff.csv', index=False)