import cv2
import os
import time

myPath = 'images/breadboard_images'

moduleVal = 10 # save every 10th frame
minBlur = 100  # smaller value means more blur is present
grayImage = False
saveData = True
showImage = True
imgWidth = 180
imgHeight = 120

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

count =0
countSave = 0

def saveDataFunc():
    global countFolder
    countFolder = 0
    while os.path.exists(myPath + str(countFolder)):
        countFolder = countFolder + 1
    os.makedirs(myPath + str(countFolder))

if saveData:saveDataFunc()

while True:
    success, img = cap.read()
    resized_frame = cv2.resize(img, (new_width, new_height))

    if grayImage:resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    if saveData:
        blur = cv2.Laplacian(resized_frame, cv2.CV_64F).var()
        if count % moduleVal == 0:
            nowTime = time.time()
            cv2.imwrite(myPath + str(countFolder) + '/'
                        + str(countSave)+"_"+str(int(blur))+"_"+
                        str(nowTime)+".png", resized_frame)
            countSave+=1
        count += 1

    cv2.imshow('Scaled Video', resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close any open windows
cap.release()
cv2.destroyAllWindows()