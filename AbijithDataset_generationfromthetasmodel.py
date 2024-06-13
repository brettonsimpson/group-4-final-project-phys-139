import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set a fixed length for all waveforms
fixed_length = 16384  # Adjust this based on your specific needs

# Path to the directory containing the files
directory = 'FINAL_DATA/'

# Function to read and preprocess data from a file
def read_data(file_path):
    data = np.loadtxt(file_path)
    if len(data) > fixed_length:
        data = data[:fixed_length]  # Truncate
    elif len(data) < fixed_length:
        data = np.pad(data, (0, fixed_length - len(data)), 'constant')  # Pad
    return data

# List all files in the directory
all_files = os.listdir(directory)

# Filter files that contain 'BBH' or 'BNS'
filtered_files = [file for file in all_files if 'BBH' in file or 'BNS' in file]

# Initialize lists for data and labels
data = []
labels = []

for file_name in filtered_files:
    file_path = os.path.join(directory, file_name)
    try:
        file_data = read_data(file_path)
        data.append([file_data])
        label = 0 if 'BBH' in file_name else 1
        labels.append(label)
    except Exception as e:
        print(f"Failed to process {file_name}: {str(e)}")

# Check if data and labels have the same length
if len(data) != len(labels):
    raise ValueError("Mismatch between the number of data samples and labels")






data = np.array(data).reshape(-1, fixed_length, 1)
labels = np.array(labels)

# One-hot encode labels
#labels_encoded = to_categorical(labels, num_classes=2)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Now you can proceed with model training as you have set it up


# One-hot encode labels
y_train_encoded = to_categorical(y_train, num_classes=2)
y_test_encoded = to_categorical(y_test, num_classes=2)
