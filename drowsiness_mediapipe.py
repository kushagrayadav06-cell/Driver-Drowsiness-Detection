import cv2
import time
import mediapipe as mp
import pygame
from math import dist

# ================== AUDIO SETUP ==================
audio_available = True

try:
    pygame.mixer.init()
except Exception as e:
    print("Audio initialization failed:", e)
    audio_available = False

if audio_available:
    pygame.mixer.music.load("alarm.wav")
    pygame.mixer.music.set_volume(1.0)

    face_warning_sound = pygame.mixer.Sound("face_warning.wav")
    face_warning_sound.set_volume(1.0)

# ================== MEDIAPIPE ==================
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ================== EYE LANDMARKS ==================
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ================== SETTINGS ==================
EAR_THRESHOLD = 0.21
DROWSY_TIME = 2.5
ALARM_DURATION = 8

FACE_MISSING_TIME = 2
FACE_WARNING_INTERVAL = 5

# ================== VARIABLES ==================
eye_closed_start = None

alarm_on = False
alarm_start = None

face_missing_start = None
last_face_warning_time = 0

# ================== FUNCTIONS ==================
def eye_aspect_ratio(eye):
    A = dist(eye[1], eye[5])
    B = dist(eye[2], eye[4])
    C = dist(eye[0], eye[3])

    return (A + B) / (2.0 * C)

# ================== CAMERA ==================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam not found!")
    exit()

print("Press Q to quit")

# ================== MAIN LOOP ==================
while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        break

    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = face_mesh.process(rgb)

    current_time = time.time()

    # ================== FACE NOT DETECTED ==================
    if not result.multi_face_landmarks:

        cv2.putText(
            frame,
            "FACE NOT DETECTED",
            (50, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        if face_missing_start is None:
            face_missing_start = current_time
            last_face_warning_time = 0

        elif current_time - face_missing_start >= FACE_MISSING_TIME:

            if current_time - last_face_warning_time >= FACE_WARNING_INTERVAL:

                if audio_available:
                    face_warning_sound.play()

                last_face_warning_time = current_time

        eye_closed_start = None

    # ================== FACE DETECTED ==================
    else:

        face_missing_start = None
        last_face_warning_time = 0

        if audio_available:
            face_warning_sound.stop()

        for face in result.multi_face_landmarks:

            landmarks = face.landmark

            left_eye = [
                (int(landmarks[i].x * w), int(landmarks[i].y * h))
                for i in LEFT_EYE
            ]

            right_eye = [
                (int(landmarks[i].x * w), int(landmarks[i].y * h))
                for i in RIGHT_EYE
            ]

            ear = (
                eye_aspect_ratio(left_eye)
                + eye_aspect_ratio(right_eye)
            ) / 2

            # ================== EAR DISPLAY ==================
            color = (0, 255, 0)

            if ear < EAR_THRESHOLD:
                color = (0, 0, 255)

            cv2.putText(
                frame,
                f"EAR: {ear:.2f}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )

            # ================== DROWSINESS ==================
            if ear < EAR_THRESHOLD:

                if eye_closed_start is None:
                    eye_closed_start = current_time

                elif current_time - eye_closed_start >= DROWSY_TIME:

                    if not alarm_on:

                        if audio_available:
                            pygame.mixer.music.play(-1)

                        alarm_on = True
                        alarm_start = current_time

            else:

                eye_closed_start = None

                if alarm_on:

                    if audio_available:
                        pygame.mixer.music.stop()

                    alarm_on = False
                    alarm_start = None

            # ================== ALARM DURATION ==================
            if alarm_on and alarm_start is not None:

                if current_time - alarm_start >= ALARM_DURATION:

                    if audio_available:
                        pygame.mixer.music.stop()

                    alarm_on = False
                    alarm_start = None
                    eye_closed_start = None

    # ================== WARNING DISPLAY ==================
    if alarm_on:

        cv2.putText(
            frame,
            "DROWSY! WAKE UP!",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )

    cv2.imshow(
        "AI Driver Drowsiness Detection System",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ================== CLEANUP ==================
cap.release()
cv2.destroyAllWindows()

if audio_available:
    pygame.mixer.quit()
