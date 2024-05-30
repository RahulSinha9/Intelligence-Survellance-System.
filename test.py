import cv2
import numpy as np
from keras.models import load_model
import winsound 
import threading
# Load the trained model
model = load_model("model.h5")

def play_alarm_sound():
    winsound.Beep(1000, 500)
# Function to predict abnormal events in a video and draw red boxes
def detect_abnormal_events(input_video_path):
    cap = cv2.VideoCapture(input_video_path)
    # cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize the frame to match the input size of the trained model
        resized_frame = cv2.resize(frame, (32,32))
        
        # Preprocess the frame for prediction
        input_frame = np.expand_dims(resized_frame, axis=0)
        input_frame = input_frame / 255.0  # Normalize pixel values
        
        # Predict using the loaded model
        prediction = model.predict(input_frame)
        # prediction[0][0] = format(prediction[0][0], '.3f')
        # prediction[0][1] = format(prediction[0][1], '.3f')
        # print(prediction[0][0])
        # print(prediction[0][1])
        # Check if abnormal event is detected (class index 1)
        if prediction[0][1] >= 0.089:  # Assuming a threshold of 0.5 for abnormal class
            # Draw a red box around the detected abnormal event
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 2)
            
            # Write "Abnormal Event" on the frame in red color
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottom_left_corner = (int(frame.shape[1] / 2) - 150, int(frame.shape[0] / 6))
            font_scale = 1
            font_color = (0, 0, 255)
            line_type = 2
            cv2.putText(frame, "Abnormal Event", bottom_left_corner, font, font_scale, font_color, line_type)

            threading.Thread(target=play_alarm_sound).start()  # You can adjust the frequency (1000) and duration (500) as needed

        display_frame = cv2.resize(frame, (640, 480))
        # Display the frame
        cv2.imshow("Video", display_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Test the function with an input video file
input_video_path = "F:/Survellience/DCSASS Dataset/Shooting/Shooting002_x264.mp4/Shooting002_x264_0.mp4"
# input_video_path = "D:/Major/DCSASS Dataset/Fighting/Fighting007_x264.mp4/Fighting007_x264_1.mp4"
# input_video_path = "D:/Major/DCSASS Dataset/Assault/Assault002_x264.mp4/Assault002_x264_4.mp4"
detect_abnormal_events(input_video_path)
