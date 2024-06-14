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
#filtered_files = [file for file in all_files if 'BBH' in file or 'BNS' in file]

# Initialize lists for data and labels
data = []
labels = []

for file_name in all_files:
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

data = np.array(data)
labels = np.array(labels)
print(data.shape)
data = np.squeeze(data, axis=1) 
#print(data)
#print(data.shape)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, ReLU, BatchNormalization, Softmax
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

# Sample data generation
# This should be replaced with your actual data loading and preprocessing
#data = np.array(data)#.reshape(-1, fixed_length, 1)

# Convert labels to numpy array and one-hot encode
#labels = to_categorical(np.array(labels), num_classes=2)

# Reshape data for the Conv1D input


# One-hot encode the labels
labels = to_categorical(labels, num_classes=2)


X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

model = Sequential([
    Conv1D(filters=32, kernel_size=3, padding='same', input_shape=(X_train.shape[1], X_train.shape[2])),
    ReLU(),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=64, kernel_size=3, padding='same'),
    ReLU(),
    BatchNormalization(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Assume model and X_test are already defined and the model is trained
# Predict probabilities
y_pred_probs = model.predict(X_test)

# Assuming y_test is already one-hot encoded and binary classification
# Extract probabilities for the positive class
y_pred_probs = y_pred_probs[:, 1]  # Change this if different setup
y_true = y_test[:, 1]  # True labels for the positive class

# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Output the AUC
print(f'AUC: {roc_auc:.2f}')
