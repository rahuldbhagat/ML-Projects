import os

data_dir = r'mnist_data'
labels_path = os.path.join(data_dir, 'train-labels.idx1-ubyte')

print(f"Full path: {os.path.abspath(labels_path)}")
print(f"File exists: {os.path.exists(labels_path)}")
print(f"File is readable: {os.access(labels_path, os.R_OK)}")

try:
    with open(labels_path, 'rb') as f:
        print("File opened successfully! First 8 bytes:", f.read(8))
except Exception as e:
    print(f"Error: {e}")