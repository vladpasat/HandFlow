import cv2
import numpy as np
import random

cap = cv2.VideoCapture(0)
success, img = cap.read()
original_height, original_width, _ = img.shape
max_width = 1920
max_height = 1080
scale_x = max_width / original_width
scale_y = max_height / original_height
scale = min(scale_x, scale_y)
new_width = int(original_width * scale)
new_height = int(original_height * scale)
cv2.namedWindow('Scaled Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Scaled Video', new_width, new_height)

cascade_path = "training/classifier/cascade.xml"  # Replace with the path to your cascade classifier XML file
cascade = cv2.CascadeClassifier(cascade_path)

while cap.isOpened():
    # Read a frame from the video stream
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    objects=cascade.detectMultiScale(gray, 10, 5)

    for (x, y , w, h) in objects:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),3)


    # Display the frame
    cv2.imshow('Scaled Video', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close any open windows
cap.release()
cv2.destroyAllWindows()





















# import cv2
#
# # Create a VideoCapture object and open the video file
# cap = cv2.VideoCapture("ex2.mp4")  # Replace "path_to_video_file.mp4" with the actual path
#
# # Check if the video file was opened successfully
# if not cap.isOpened():
#     print("Error opening video file")
#     exit()
#
#
#
# # Read and display frames from the video
# while True:
#     # Read a frame from the video
#     ret, frame = cap.read()
#
#     # Check if the frame was read successfully
#     if not ret:
#         break
#
#     # Display the frame in a window named "Video"
#     cv2.imshow("Video", frame)
#
#     # Break the loop if the 'q' key is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release the VideoCapture object and close the window
# cap.release()
# cv2.destroyAllWindows()