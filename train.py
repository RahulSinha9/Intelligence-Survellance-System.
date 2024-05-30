import os
import cv2
import math
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical

# Define constants
NUM_CLASSES = 2  # Abnormal and Normal (change this based on your dataset)
IMG_SIZE = (32, 32)  # Resize frames to 64x64 for consistency

# Function to load and preprocess video frames and labels
def load_dataset(dataset_path, labels_path):
    X = []
    y = []
    
    flag = 0
    # Iterate over class folders
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        
        # Read CSV file for labels
        labels_file_path = os.path.join(labels_path, f"{class_name}.csv")
        labels_df = pd.read_csv(labels_file_path)
        
        # Iterate over subclass folders
        for subclass_name in os.listdir(class_path):
            subclass_path = os.path.join(class_path, subclass_name)
            
            # Iterate over video files in the current subclass
            for video_file in os.listdir(subclass_path):
                if video_file.endswith('.mp4'):
                    video_path = os.path.join(subclass_path, video_file)
                    
                    # Read video frames and resize to IMG_SIZE
                    cap = cv2.VideoCapture(video_path)
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame = cv2.resize(frame, IMG_SIZE)
                        X.append(frame)
                        
                        # Extract label from CSV file based on frame index
                        # frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                        if flag < len(labels_df):
                            label = labels_df.iloc[flag][2]  # Assuming the label column in CSV is "Label"
                            
                            # Check for NaN values and handle them (assign a default label or skip the frame)
                            if not math.isnan(label):
                                y.append(int(label))  # 1 for abnormal, 0 for normal
                            else:
                                y.append(0)
                        else:
                            y.append(0)
                    flag = flag + 1
                    cap.release()
                    
    X = np.array(X)
    y = to_categorical(y, NUM_CLASSES)
    return X, y


# Load the dataset
dataset_path = "F:/Survellience/DCSASS Dataset"
labels_path = "F:/Survellience/Labels"
X, y = load_dataset(dataset_path, labels_path)

# Build a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(NUM_CLASSES, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
# Train the model on the entire dataset
# model.fit(X, y, epochs=10, batch_size=32)

# Save the trained model to a file
model.save("model.h5")

print("Model saved as model.h5")
