import pdb
import numpy as np
import pandas as pd
import os

def read_and_preprocess(target_dir, filepath, idx, time_label, domain_label):
    # Read data, skipping the header and converting data to appropriate types
    data = pd.read_csv(filepath, sep="\t", header=0)
    data['time'] = pd.to_numeric(data['time'], errors='coerce')
    # pdb.set_trace()
    for col in data.columns[1:-1]:  # Skip Time and Class columns
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data['class'] = pd.to_numeric(data['class'], downcast='integer', errors='coerce')

    # pdb.set_trace()
    data = data[data['class'] != 0]
    data = data[data['class'] != 7]
    
    data['Time_diff'] = data['time'].diff().fillna(0).abs()  # Fill initial NA in diff and take absolute value
    threshold = 10  # Define a threshold for considering breaks in continuity (e.g., 10 ms)
    data['Time_group'] = (data['Time_diff'] > threshold).cumsum()  # Increment group id when a break is detected


    window_size = 52  # Number of samples per segment
    # pdb.set_trace()
    # Process each continuous segment
    for _, group in data.groupby('Time_group'):
        # Check if the group size is at least the window size to make at least one segment
        group.set_index('time', inplace=True)
        group = group.reindex(range(group.index[0], group.index[-1] + 1, 5), method='nearest').dropna()

        if len(group) >= window_size:
            # Calculate how many full segments we can get from this group
            num_segments = len(group) // window_size
            for i in range(num_segments):
                start = i * window_size
                end = start + window_size
                segment = group.iloc[start:end, 1:-2].values  # Exclude Time, Class, and Time_diff columns
                segment = np.expand_dims(segment, axis=0)
                label = group['class'].iloc[start]
                
                file_name = f"{target_dir}/{idx}.npz"
                # pdb.set_trace()
                np.savez(file_name, emg=segment, add_infor=np.array([label-1, domain_label, time_label]))

                idx += 1
                time_label += 1
                print(time_label)
        
        time_label += 2000

    return idx, time_label


def read_and_preprocess_cda(target_dir, filepath, idx, time_label, domain_label):
    # Read data, skipping the header and converting data to appropriate types
    data = pd.read_csv(filepath, sep="\t", header=0)
    data['time'] = pd.to_numeric(data['time'], errors='coerce')
    # pdb.set_trace()
    for col in data.columns[1:-1]:  # Skip Time and Class columns
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data['class'] = pd.to_numeric(data['class'], downcast='integer', errors='coerce')

    # pdb.set_trace()
    data = data[data['class'] != 0]
    data = data[data['class'] != 7]
    
    data['Time_diff'] = data['time'].diff().fillna(0).abs()  # Fill initial NA in diff and take absolute value
    threshold = 10  # Define a threshold for considering breaks in continuity (e.g., 10 ms)
    data['Time_group'] = (data['Time_diff'] > threshold).cumsum()  # Increment group id when a break is detected


    window_size = 52  # Number of samples per segment
    # pdb.set_trace()
    # Process each continuous segment
    for _, group in data.groupby('Time_group'):
        # Check if the group size is at least the window size to make at least one segment
        group.set_index('time', inplace=True)
        group = group.reindex(range(group.index[0], group.index[-1] + 1, 5), method='nearest').dropna()

        if len(group) >= window_size:
            # Calculate how many full segments we can get from this group
            num_segments = len(group) // window_size
            for i in range(num_segments):
                start = i * window_size
                end = start + window_size
                segment = group.iloc[start:end, 1:-2].values  # Exclude Time, Class, and Time_diff columns
                label = group['class'].iloc[start]
                
                file_name = f"{target_dir}/{idx}.npz"
                # pdb.set_trace()
                np.savez(file_name, emg=segment, add_infor=np.array([label-1, domain_label, time_label]))

                idx += 1
                time_label += 1
                print(time_label)
        
        time_label += 2000

    return idx, time_label

def process_folder(folder_path, cda=False):
    target_dir = "./datasets/UCI"
    # pdb.set_trace()
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    user_folders = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    time_label = 0
    idx = 0
    for domain_label, user_folder in enumerate(user_folders):
        files = [os.path.join(user_folder, f) for f in os.listdir(user_folder) if f.endswith('.txt')]
        for file in files:
            if cda:
                idx, time_label = read_and_preprocess_cda(target_dir, file, idx, time_label, domain_label)
            else:
                idx, time_label = read_and_preprocess(target_dir, file, idx, time_label, domain_label)
    

# Example usage
process_folder('./original_dataset/UCI_EMG')
# process_folder('./original_dataset/UCI_EMG', cda=True)