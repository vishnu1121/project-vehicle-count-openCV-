import cv2
import numpy as np
from time import sleep

min_width_rect = 80  # width of the rectangle
min_hight_rect = 80  # height of the rectangle

offset = 6

count_line_pos = 550  # Count Line Position

delay = 15  # FPS do vídeo
detec = []
count_vehicle = 0  # Count Of vehicle


def central_handle(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


cap = cv2.VideoCapture('video.mp4')
# background Segementation
subtracao = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret, frame1 = cap.read()
    speed_of_video = float(1/delay)
    sleep(speed_of_video)
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # Convert Color
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    img_sub = subtracao.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    shape = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (5, 5))  # circle shape
    dialated_img = cv2.morphologyEx(dilat, cv2. MORPH_CLOSE, shape)
    dialated_img = cv2.morphologyEx(dialated_img, cv2. MORPH_CLOSE, shape)
    find_Contours, h = cv2.findContours(
        dialated_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1, (25, count_line_pos),
             (1200, count_line_pos), (0, 0, 255), 3)

    for(i, c) in enumerate(find_Contours):
        (x, y, w, h) = cv2.boundingRect(c)
        valid_contorno = (w >= min_width_rect) and (h >= min_hight_rect)
        if not valid_contorno:
            continue

        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame1, "Vehicle"+str(count_vehicle), (x, y-20),
                    cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 244, 0), 2)

        center = central_handle(x, y, w, h)
        detec.append(center)
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)

        # loop
        for (x, y) in detec:
            if y < (count_line_pos+offset) and y > (count_line_pos-offset):
                count_vehicle += 1
                cv2.line(frame1, (25, count_line_pos),
                         (1200, count_line_pos), (0, 255, 0), 3)
                detec.remove((x, y))
                print("car is detected : "+str(count_vehicle))

    cv2.putText(frame1, "VEHICLE COUNT : "+str(count_vehicle), (450, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)
    cv2.imshow("Video Original", frame1)

    if cv2.waitKey(1) == 13:
        break

cv2.destroyAllWindows()
cap.release()
