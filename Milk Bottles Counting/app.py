import cv2 as cv
from ultralytics import YOLO
import math
import cvzone
import numpy as np
from flask import Flask, render_template, url_for, Response
from Sort import *

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/inspect')
def inspect():
    return render_template('inspect.html')

@app.route('/video_feed')
def video_feed():
    return Response(bottle_detects(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Open the video file named 'milk_bottles.mp4' for capturing frames
video_path = 'C:/Users/smart/Downloads/milk_bottles.mp4'
cap = cv.VideoCapture(video_path)  
if not cap.isOpened():
    print("Error: Unable to open video file '{}'.".format(video_path))
    exit()

# Load the YOLO object detection model from 'yolov8n.pt'
model = YOLO('yolov8n.pt')

# Read class names from the 'classes.txt' file
classnames = []
with open('classes.txt','r') as f:
    classnames = f.read().splitlines()
print(classnames[0])

# Initialize the SORT tracker with specified parameters
tracker = Sort(max_age=20, min_hits=3)

# Define the coordinates of a line for counting objects 
line = [100, 0, 100, 900]

# Initialize an empty list to store IDs of objects that crossed the line
counter = []

# Loop to process video frames and perform object detection and tracking
def bottle_detects():
    while True:
        ret, video = cap.read()  # Read a frame from the video capture
        if not ret:
            print("Error: Unable to read frame from video.")
            break
        
        detections = np.empty((0, 5))  # Initialize an empty NumPy array to store object detections
        
        # Perform object detection on the current frame using YOLO
        results = model(video, stream=1)
        for info in results:  # Iterate over the detected objects
            parameters = info.boxes  # Extract the bounding box parameters of each object
            for details in parameters:  # Iterate over the details of each detected object
                x1, y1, x2, y2 = details.xyxy[0]  # Extract the coordinates of the bounding box
                conf = details.conf[0]  # Extract the confidence score of the detection
                conf = math.ceil(conf * 100)  # Convert the confidence score to percentage
                class_detect = details.cls[0]  # Extract the class index of the detection
                class_detect = int(class_detect)  # Convert the class index to an integer
                class_detect = classnames[class_detect]  # Get the class name from the list
                
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert coordinates to integers
                current_detections = np.array([x1, y1, x2, y2, conf])  # Create an array of detection parameters
                detections = np.vstack((detections, current_detections))  # Append the detection to the array
        
        results = tracker.update(detections)  # Update the SORT tracker with the detected objects
        cv.line(video, (line[0], line[1]), (line[2], line[3]), (0, 255, 255), 5)  # Draw a line on the video frame
        
        for info in results:  # Iterate over the tracked objects
            x1, y1, x2, y2, id = info  # Extract the coordinates and ID of each tracked object
            x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)  # Convert coordinates and ID to integers
            w, h = x2 - x1, y2 - y1  # Calculate the width and height of the bounding box
            cx, cy = x1 + w // 2, y1 + h // 2  # Calculate the center coordinates of the bounding box
            
            cv.circle(video, (cx, cy), 12, (255, 0, 255), -1)  # Draw a filled circle at the center of the bounding box
            cvzone.cornerRect(video, [x1, y1, w, h], rt=5)  # Draw a rectangle around the object
            if line[1] < cy < line[3] and line[2] - 10 < cx < line[2] + 10:  # Check if the object crossed the line
                cv.line(video, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 10)  # Draw a thicker line
                if id not in counter:  # If the object ID is not in the counter list
                    counter.append(id)  # Add the object ID to the counter list
        
            cvzone.putTextRect(video, f'Milk Bottles Count = {len(counter)}', [200, 34],colorT=(255,0,0), colorB=(0,0,255),
                            thickness=1, scale=1.5, border=1)  # Add text displaying the total count of objects
            ret, buffer = cv.imencode('.jpg', video)# Encode the frame as JPEG
            frame = buffer.tobytes()  # Convert the frame to bytes
            yield (b'--frame\r\n'  # Yield the frame as part of a multipart response
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
if __name__ == '__main__':
    app.run(debug=True)
