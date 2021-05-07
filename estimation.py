import os,sys
from scipy.spatial import distance
import numpy as np
import cv2
import dlib
from imutils import face_utils

cap = cv2.VideoCapture(0)
#f = int(cap.get(cv2.CAP_PROP_FPS))
#w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
#writer = cv2.VideoWriter('output.mp4', fourcc, f, (w, h))
face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_alt2.xml')
face_parts_detector = dlib.shape_predictor('./data/shape_predictor_68_face_landmarks.dat')
cvFont = cv2.FONT_HERSHEY_PLAIN
model_points = np.array([
        (0.0,0.0,0.0), # 30
        (-30.0,-125.0,-30.0), # 21
        (30.0,-125.0,-30.0), # 22
        (-60.0,-70.0,-60.0), # 39
        (60.0,-70.0,-60.0), # 42
        (-40.0,40.0,-50.0), # 31
        (40.0,40.0,-50.0), # 35
        (-70.0,130.0,-100.0), # 48
        (70.0,130.0,-100.0), # 54
        (0.0,158.0,-10.0), # 57
        (0.0,250.0,-50.0) # 8
        ])
size = (720, 1280)
focal_length = size[1]
center = (size[1] // 2, size[0] // 2)
camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype='double')
dist_coeffs = np.zeros((4, 1))

def calc_ear(eye_c):
    a = distance.euclidean(eye_c[1], eye_c[5])
    b = distance.euclidean(eye_c[2], eye_c[4])
    c = distance.euclidean(eye_c[0], eye_c[3])
    ear = (a + b) / (c * 2.0)
    return round(ear, 3)

def detect_eye_close(right_ear, left_ear):
    if ((right_ear < 0.25) and (left_ear <0.25)):
        return True
    else:
        return False

def show_params(eye_state, tick, frame, blink_count):
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - tick)
    cv2.putText(frame, "FPS:{} ".format(int(fps)), (10, 50),
        cvFont, 3, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Blink Count:{} ".format(blink_count), (10, 80),
            cvFont, 2, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Eye:{} ".format(eye_state), (10, 120),
            cvFont, 2, (0, 0, 0), 2, cv2.LINE_AA)

def main():
    blink_count = 0
    eye_state = "OPEN"
    pre_is_eye_closed = False

    while True:
        tick = cv2.getTickCount()

        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.11, minNeighbors=1, minSize=(100, 100))

        if len(faces) == 1:
            x, y, w, h = faces[0, :]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            face = dlib.rectangle(x, y, x + w, y + h)
            face_parts = face_parts_detector(gray, face)
            face_parts = face_utils.shape_to_np(face_parts)

            parts_for_estimation = np.array([
                (face_parts[30]),
                (face_parts[21]),
                (face_parts[22]),
                (face_parts[39]),
                (face_parts[42]),
                (face_parts[31]),
                (face_parts[35]),
                (face_parts[48]),
                (face_parts[54]),
                (face_parts[57]),
                (face_parts[8]),
                ],dtype='double')

            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, parts_for_estimation, camera_matrix,dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 500.0)]), rotation_vector,translation_vector, camera_matrix, dist_coeffs)
            right_ear = calc_ear(face_parts[42:48])
            left_ear = calc_ear(face_parts[36:42])

            is_eye_closed = detect_eye_close(right_ear, left_ear)

            if (is_eye_closed and (is_eye_closed != pre_is_eye_closed)):
                blink_count+=1

            pre_is_eye_closed = is_eye_closed

            for i, ((x, y)) in enumerate(face_parts):
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                #cv2.putText(frame, str(i), (x + 2, y - 2),cvFont, 0.3, (0, 255, 0), 1)

            p1 = (int(parts_for_estimation[0][0]), int(parts_for_estimation[0][1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

            cv2.arrowedLine(frame, p1, p2, (255, 0, 0), 2)
        else:
            cv2.putText(frame, "Face Detection Failed", (10, 160),
                    cvFont, 2, (0, 0, 255), 2, cv2.LINE_AA)

        if (is_eye_closed):
            eye_state = "CLOSE"
        else:
            eye_state = "OPEN"

        show_params(eye_state, tick, frame, blink_count)
        cv2.imshow('frame', frame)
        #writer.write(frame)

        if cv2.waitKey(1) == 27:
            break


if __name__ == '__main__':
    main()
    cap.release()
    cv2.destroyAllWindows()
