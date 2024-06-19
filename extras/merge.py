import pandas as pd

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

csv_files = [('Hull1.csv','hull1'),('Hull2.csv', 'hull2'),('Hull3.csv', 'hull3'),('Hull4.csv', 'hull4'), ('CGAL.csv', 'CGAL')]
parse_csv_and_merge(csv_files)
