import cv2
import mediapipe as mp
from handLandmarksDefine import *
from objectCoords import *
import time
import numpy as np
import math
import serial

start_time = 0
end_time = 0

try:
    arduino = serial.Serial('COM3', 9600)
except serial.serialutil.SerialException:
    print("Arduino not connected")


# lm = landmark
class HandLandmarkDetector:
    def __init__(self, static_image_mode, max_num_hands, model_complexity, min_detection_confidence,
                 min_tracking_confidence):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.static_image_mode, self.max_num_hands, self.model_complexity,
                                        self.min_detection_confidence, self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None

    def draw_hand_landmarks(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        if self.results.multi_hand_landmarks:
            for self.handLMS in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, self.handLMS, self.mpHands.HAND_CONNECTIONS)
        return img

    def find_hand_landmark_coordinates(self, img):
        landmark_list = []
        if self.results.multi_hand_landmarks:
            for id, lm in enumerate(self.handLMS.landmark):
                print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmark_list.append([id, cx, cy])
                cv2.circle(img, (cx, cy), 7, (255, 255, 255), cv2.FILLED)
        return landmark_list


class HandTracker:
    def __init__(self, width, height, top_left_x, top_left_y, color, thickness):
        self.width = width
        self.height = height
        self.top_left_x = top_left_x
        self.top_left_y = top_left_y
        self.color = color
        self.thickness = thickness
        self.object_assembled = False

    def reset_sizes(self, width, height, top_left_x, top_left_y, color, thickness):
        self.width = width
        self.height = height
        self.top_left_x = top_left_x
        self.top_left_y = top_left_y
        self.color = color
        self.thickness = thickness

    def draw(self, img):
        cv2.rectangle(img, (self.top_left_x, self.top_left_y), (self.top_left_x + self.width, self.top_left_y +
                                                                self.height), self.color, self.thickness)

    def set_green_color(self):
        self.color = (0, 255, 0)
        self.object_assembled = True

    def set_red_color(self):
        self.color = (0, 0, 255)
        self.object_assembled = False

    def set_object_assembled_false(self):
        self.object_assembled = False

    def set_object_assembled_true(self):
        self.object_assembled = True

    def detect_hand_inside_area(self, landmark_list):
        global start_time
        global end_time
        if len(landmark_list) != 0 and self.object_assembled is False:
            if self.top_left_x <= landmark_list[INDEX_FINGER_TIP][1] <= self.top_left_x + self.width and self.top_left_y <= landmark_list[INDEX_FINGER_TIP][2] <= self.top_left_y + self.height\
                    and self.top_left_x <= landmark_list[THUMB_TIP][1] <= self.top_left_x + self.width and self.top_left_y <= landmark_list[THUMB_TIP][2] <= self.top_left_y + self.height:
                self.color = (0, 255, 0)
                if start_time == 0 and end_time == 0:
                    start_time = time.time()
                    end_time = start_time + 3
                    # print("end time: ")
                    # print(end_time)
                    # print("current time: ")
                    # print(time.time())
                else:
                    if time.time() > end_time:
                        start_time = 0
                        end_time = 0
                        return 1
            else:
                self.color = (0, 0, 255)

    def gesture_control(self, landmark_list, resized_frame, arduino):
        global start_time
        global end_time
        if len(landmark_list) != 0:
            x1, y1 = landmark_list[THUMB_TIP][1], landmark_list[THUMB_TIP][2]
            x2, y2 = landmark_list[INDEX_FINGER_TIP][1], landmark_list[INDEX_FINGER_TIP][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(resized_frame, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(resized_frame, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(resized_frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(resized_frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            length = math.hypot(x2 - x1, y2 - y1)
            vol = np.interp(length, [70, 250], [0, 250])
            if start_time == 0 and end_time == 0:
                start_time = time.time()
                end_time = start_time + 1
                # print("end time: ")
                # print(end_time)
                # print("current time: ")
                # print(time.time())
            else:
                if time.time() > end_time:
                    start_time = 0
                    end_time = 0
                arduino.write(str(vol).encode())


class ObjectAssembler:
    def draw_work_area(self, resized_frame, work_area_top_left, work_area_bottom_right, work_area_color,
                       work_area_thickness, text_content, text_font, text_font_scale, text_color, text_thickness):
        cv2.rectangle(resized_frame, work_area_top_left, work_area_bottom_right, work_area_color, work_area_thickness)
        text_x = work_area_top_left[0] + 5
        text_y = work_area_top_left[1] + 30
        cv2.putText(resized_frame, text_content, (text_x, text_y), text_font, text_font_scale, text_color,
                    text_thickness)

    def draw_component_area(self, resized_frame,component_area_top_left, component_area_bottom_right,
                            component_area_color, component_area_thickness, text_content, text_font, text_font_scale,
                            text_color, text_thickness):
        cv2.rectangle(resized_frame, component_area_top_left, component_area_bottom_right, component_area_color,
                      component_area_thickness)
        text_x = component_area_top_left[0] + 5
        text_y = component_area_top_left[1] + 30
        cv2.putText(resized_frame, text_content, (text_x, text_y), text_font, text_font_scale, text_color,
                    text_thickness)

    def draw_breadboard_outline(self, resized_frame, breadboard_top_left, breadboard_bottom_right,
                                breadboard_outline_color, breadboard_outline_thickness):
        cv2.rectangle(resized_frame, breadboard_top_left, breadboard_bottom_right, breadboard_outline_color,
                      breadboard_outline_thickness)

    def draw_next_component(self, resized_frame, next_top_left, next_bottom_right, next_color, next_thickness,
                            text_content, text_font, text_font_scale, text_color, text_thickness):
        cv2.rectangle(resized_frame, next_top_left, next_bottom_right, next_color, next_thickness)
        text_x = next_top_left[0] + 40
        text_y = next_top_left[1] + 500
        cv2.putText(resized_frame, text_content, (text_x, text_y), text_font, text_font_scale, text_color,
                    text_thickness)

    def draw_previous_component(self, resized_frame, previous_top_left, previous_bottom_right, previous_color,
                                previous_thickness, text_content, text_font, text_font_scale, text_color,
                                text_thickness):
        cv2.rectangle(resized_frame, previous_top_left, previous_bottom_right, previous_color, previous_thickness)
        text_x = previous_top_left[0] + 10
        text_y = previous_top_left[1] + 500
        cv2.putText(resized_frame, text_content, (text_x, text_y), text_font, text_font_scale, text_color,
                    text_thickness)

    def detect_finger_inside_next_component(self, landmark_list, next_top_left, next_bottom_right):
        global start_time
        global end_time
        if len(landmark_list) != 0:
            if next_top_left[0] <= landmark_list[PINKY_TIP][1] <= next_bottom_right[0] and \
                    next_top_left[1] <= landmark_list[PINKY_TIP][2] <= next_bottom_right[1]:
                if start_time == 0 and end_time == 0:
                    start_time = time.time()
                    end_time = start_time + 3
                else:
                    if time.time() > end_time:
                        start_time = 0
                        end_time = 0
                        return 1

    def detect_finger_inside_previous_component(self, landmark_list, previous_top_left, previous_bottom_right):
        global start_time
        global end_time
        if len(landmark_list) != 0:
            if previous_top_left[0] <= landmark_list[THUMB_TIP][1] <= previous_bottom_right[0] and \
                    previous_top_left[1] <= landmark_list[THUMB_TIP][2] <= previous_bottom_right[1]:
                if start_time == 0 and end_time == 0:
                    start_time = time.time()
                    end_time = start_time + 3
                else:
                    if time.time() > end_time:
                        start_time = 0
                        end_time = 0
                        return 1


def resize_window(img, max_width, max_height):
    original_height, original_width, _ = img.shape
    scale_x = max_width / original_width
    scale_y = max_height / original_height
    scale = min(scale_x, scale_y)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    return new_width, new_height


def main():
    cap = cv2.VideoCapture(0)
    success, img = cap.read()
    new_width, new_height = resize_window(img, max_width=1680, max_height=1050)
    cv2.namedWindow('Scaled Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Scaled Video', new_width, new_height)
    cv2.setWindowProperty('Scaled Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    hand_detector = HandLandmarkDetector(static_image_mode=False, max_num_hands=1, model_complexity=1,
                                         min_detection_confidence=0.1, min_tracking_confidence=0.1)
    i = 0
    gestureControl = False
    
    rectangle_width = [ARDUINO_WIDTH, LED_WIDTH, LED_BAR_WIDTH]
    rectangle_height = [ARDUINO_HEIGHT, LED_HEIGHT, LED_BAR_HEIGHT]
    rectangle_top_left_x = [ARDUINO_X, LED_X, LED_BAR_X]
    rectangle_top_left_y = [ARDUINO_Y, LED_Y, LED_BAR_Y]
    rectangle_color = (0, 0, 255)
    rectangle_thickness = 2

    hand_tracker = HandTracker(rectangle_width[0], rectangle_height[0], rectangle_top_left_x[0],
                               rectangle_top_left_y[0], rectangle_color, rectangle_thickness)

    component_name = ["Arduino", "LED to PIN D9", "LED bar", "LED dimming"]

    object_assembler = ObjectAssembler()

    while True:
        success, img = cap.read()
        resized_frame = cv2.resize(img, (new_width, new_height))
        hand_tracker.draw(resized_frame)

        resized_frame = hand_detector.draw_hand_landmarks(resized_frame)
        landmark_list = hand_detector.find_hand_landmark_coordinates(resized_frame)
        
        if hand_tracker.detect_hand_inside_area(landmark_list) == 1:
            hand_tracker.set_green_color()
            hand_tracker.set_object_assembled_true()

        object_assembler.draw_previous_component(resized_frame=resized_frame, previous_top_left=(1, 1),
                                                 previous_bottom_right=(146, 1061), previous_color=(0, 0, 255),
                                                 previous_thickness = 20, text_content="Previous",
                                                 text_font=cv2.FONT_HERSHEY_SIMPLEX, text_font_scale=1.0,
                                                 text_color=(0, 0, 255), text_thickness=2)

        object_assembler.draw_next_component(resized_frame=resized_frame, next_top_left=(1250, 1),
                                             next_bottom_right=(1395, 1061), next_color=(0, 255, 0), next_thickness=20,
                                             text_content="Next", text_font=cv2.FONT_HERSHEY_SIMPLEX,
                                             text_font_scale=1.0, text_color=(0, 255, 0), text_thickness=2)

        object_assembler.draw_work_area(resized_frame=resized_frame, work_area_top_left=(145, 300),
                                        work_area_bottom_right=(1245, 1050), work_area_color=(255, 255, 255),
                                        work_area_thickness=3, text_content="Work Area",
                                        text_font=cv2.FONT_HERSHEY_SIMPLEX, text_font_scale=1.0,
                                        text_color=(255, 255, 255), text_thickness=2)

        object_assembler.draw_breadboard_outline(resized_frame=resized_frame, breadboard_top_left=(550, 400),
                                                 breadboard_bottom_right=(850, 850),
                                                 breadboard_outline_color=(0, 0, 255), breadboard_outline_thickness=2)

        object_assembler.draw_component_area(resized_frame=resized_frame, component_area_top_left=(145, 1),
                                             component_area_bottom_right=(1245, 295),
                                             component_area_color=(255, 255, 255), component_area_thickness=2,
                                             text_content="Component Area", text_font=cv2.FONT_HERSHEY_SIMPLEX,
                                             text_font_scale=1.0, text_color=(255, 255, 255), text_thickness=2)

        if object_assembler.detect_finger_inside_next_component(landmark_list, next_top_left=(1250, 1),
                                                                next_bottom_right=(1395, 1061)) == 1:
            hand_tracker.set_object_assembled_false()
            if i < 2:
                i = i + 1
                hand_tracker.reset_sizes(rectangle_width[i], rectangle_height[i], rectangle_top_left_x[i],
                                        rectangle_top_left_y[i], rectangle_color, rectangle_thickness)
            else:
                gestureControl = True

        if gestureControl == True:
            i=3
            try:
                hand_tracker.gesture_control(landmark_list, resized_frame, arduino)
            except UnboundLocalError:
                cv2.putText(resized_frame, "Connect the Arduino and restart!", (170, 400), cv2.FONT_HERSHEY_SIMPLEX,
                            2.0, (0, 0, 255), 6)
            except serial.serialutil.SerialException:
                cv2.putText(resized_frame, "Connect the Arduino and restart!", (170, 400), cv2.FONT_HERSHEY_SIMPLEX,
                            2.0, (0, 0, 255), 6)
            except NameError:
                cv2.putText(resized_frame, "Connect the Arduino and restart!", (170, 400), cv2.FONT_HERSHEY_SIMPLEX,
                            2.0, (0, 0, 255), 6)

        if object_assembler.detect_finger_inside_previous_component(landmark_list, previous_top_left=(1, 1),
                                                                    previous_bottom_right=(146, 1061)) == 1:
            hand_tracker.set_object_assembled_false()
            gestureControl = False
            if i == 0:
                i = 2
            else:
                i = i - 1
            hand_tracker.reset_sizes(rectangle_width[i], rectangle_height[i], rectangle_top_left_x[i],
                                     rectangle_top_left_y[i], rectangle_color, rectangle_thickness)

        if hand_tracker.object_assembled == True:
            cv2.putText(resized_frame, component_name[i], (650, 280), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
        else: cv2.putText(resized_frame, component_name[i], (650, 280), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)

        cv2.imshow('Scaled Video', resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()

# counter = 1
# a = 0
# b = 1
# print(a)
# print(b)
#
# while counter < 10:
#     c = a+b
#     a = b
#     b = c
#     print(c)
#     counter = counter + 1