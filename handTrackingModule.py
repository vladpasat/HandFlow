import cv2
import mediapipe as mp
import time

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
        cv2.rectangle(img, (self.top_left_x, self.top_left_y),
                      (self.top_left_x + self.width, self.top_left_y + self.height), self.color, self.thickness)


    def detectHandInsideArea(self, lmList):
        if len(lmList) != 0:
            if self.top_left_x <= lmList[4][1] <= self.top_left_x + self.width and self.top_left_y <= lmList[4][2] <= self.top_left_y + self.height:
                self.color = (0,255,0)
                #return 1
            else: self.color = (0,0,255)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1680)  # window width
    cap.set(4, 1050)  # window height
    detector = handDetector()
    rectangle_width = 300
    rectangle_height = 200
    rectangle_top_left_x = 0
    rectangle_top_left_y = 0
    rectangle_color = (0, 0, 255)  # red default
    rectangle_thickness = 5
    drawRec = drawClass(rectangle_width, rectangle_height, rectangle_top_left_x, rectangle_top_left_y, rectangle_color, rectangle_thickness)
    while True:
        success, img = cap.read()
        drawRec.draw(img)
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if drawRec.detectHandInsideArea(lmList) == 1:
            drawRec.resetSizes(400,600,30,50,rectangle_color,rectangle_thickness)

        # if len(lmList) !=0:
        # if(lmList[4][1] > 1000 and  lmList[4][2]>500):
        # print(lmList[4][1], lmList[4][2]) #list item number is the hand landmark position

        cv2.imshow("Webcam", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
