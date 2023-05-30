import cv2
import mediapipe as mp
from handLandmarksDefine import *
import time
from tracker import *

tracker = EuclideanDistTracker()

start_time = 0
end_time = 0

# lm = landmark
class handDetector():
    def __init__(self, static_image_mode=False, max_num_hands=1, model_complexity=1, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence


        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.static_image_mode, self.max_num_hands, self.model_complexity,
                                        self.min_detection_confidence, self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for self.handLMS in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, self.handLMS, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img):
        lmList = []
        if self.results.multi_hand_landmarks:
            for id, lm in enumerate(self.handLMS.landmark):
                # print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # if (id == 4):
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                cv2.circle(img, (cx, cy), 7, (255, 255, 255), cv2.FILLED)
        return lmList


class drawClass():
    def __init__(self, width, height, top_left_x, top_left_y, color, thickness):
        self.width = width
        self.height = height
        self.top_left_x = top_left_x
        self.top_left_y = top_left_y
        self.color = color
        self.thickness = thickness

    def resetSizes(self, width, height, top_left_x, top_left_y, color, thickness):
        self.width = width
        self.height = height
        self.top_left_x = top_left_x
        self.top_left_y = top_left_y
        self.color = color
        self.thickness = thickness

    def draw(self, img):
        cv2.rectangle(img, (self.top_left_x, self.top_left_y), (self.top_left_x + self.width, self.top_left_y + self.height), self.color, self.thickness)


    def detectHandInsideArea(self, lmList, interestPointsOnHand):
        global start_time
        global end_time
        if len(lmList) != 0:
            if self.top_left_x <= lmList[INDEX_FINGER_TIP][1] <= self.top_left_x + self.width and self.top_left_y <= lmList[INDEX_FINGER_TIP][2] <= self.top_left_y + self.height\
                    and self.top_left_x <= lmList[MIDDLE_FINGER_TIP][1] <= self.top_left_x + self.width and self.top_left_y <= lmList[MIDDLE_FINGER_TIP][2] <= self.top_left_y + self.height\
                    and self.top_left_x <= lmList[THUMB_TIP][1] <= self.top_left_x + self.width and self.top_left_y <= lmList[THUMB_TIP][2] <= self.top_left_y + self.height:
                self.color = (0, 255, 0)
                if(start_time == 0 and end_time == 0):
                    start_time = time.time()
                    end_time = start_time + 3
                    print("end time: ")
                    print(end_time)
                    print("current time: ")
                    print(time.time())
                else:
                    if time.time() > end_time:
                        start_time = 0
                        end_time = 0
                        return 1

            else:
                self.color = (0,0,255)

class objectDetector():
    def __init__(self, lower_threshold, upper_threshold):










def main():
    cap = cv2.VideoCapture("example.mp4")
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


    detector = handDetector()
    i=0
    rectangle_width = [200, 200, 200, 100]
    rectangle_height = [200, 200, 200, 100]
    rectangle_top_left_x = [100, 500, 800, 1000]
    rectangle_top_left_y = [100, 200, 60, 800]
    rectangle_color = (0, 0, 255)
    rectangle_thickness = 5

    interestPointsOnHand = [5, 6, 7, 8, 9, 10, 11, 12]

    objectDetect = objectDetector()

    drawRec = drawClass(rectangle_width[0], rectangle_height[0], rectangle_top_left_x[0], rectangle_top_left_y[0], rectangle_color, rectangle_thickness)
    while True:
        success, img = cap.read()
        resized_frame = cv2.resize(img, (new_width, new_height))
        drawRec.draw(resized_frame)
        resized_frame = detector.findHands(resized_frame)
        lmList = detector.findPosition(resized_frame)
        if drawRec.detectHandInsideArea(lmList, interestPointsOnHand) == 1:
            if(i == 2):
                i=0
            else:
                i = i+1
            drawRec.resetSizes(rectangle_width[i],rectangle_height[i],rectangle_top_left_x[i],rectangle_top_left_y[i], rectangle_color, rectangle_thickness)

        # if len(lmList) !=0:
        # if(lmList[4][1] > 1000 and  lmList[4][2]>500):
        # print(lmList[4][1], lmList[4][2]) #list item number is the hand landmark position


        cv2.imshow('Scaled Video', resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
