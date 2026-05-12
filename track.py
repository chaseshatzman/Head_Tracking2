import math


import cv2
import mediapipe as mp
import numpy as np


video = cv2.VideoCapture(0)


face_mesh = mp.solutions.face_mesh


face_mesh_params = face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.75, min_tracking_confidence=0.75)


drawing = mp.solutions.drawing_utils
drawing_spec = drawing.DrawingSpec(thickness=1, circle_radius=1)
X_THRESHOLD = 0.01
Y_THRESHOLD = 0.01
# Depth score uses neutral band around calibrated face size (see width_reference below)
Z_ENTER_THRESHOLD = 0.04
EMA_ALPHA = 0.25
NEUTRAL_SIZE_TOLERANCE = 0.05
# Planar-only: ignore tiny nose motion when depth is neutral (same scale as sx/sy)
PLANAR_IDLE_MAX = 0.35


calibrated = False
neutral_nose_x = 0.0
neutral_nose_y = 0.0
neutral_face_width = 0.0
neutral_head_height = 0.0
smooth_face_width = None
smooth_face_height = None


key = 255


while True:
   ret, image = video.read()
   if not ret:
       break
  
   image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   results = face_mesh_params.process(image_rgb)


   if results.multi_face_landmarks:
       cv2.putText(image, "Head detected", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


       for face_landmarks in results.multi_face_landmarks:
           drawing.draw_landmarks(image=image, landmark_list=face_landmarks, connections=face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=drawing_spec, connection_drawing_spec=drawing_spec)


           lm = face_landmarks.landmark
           nose = lm[1]
           nose_x = nose.x
           nose_y = nose.y
           # Cheek-to-cheek span grows when you move toward the camera, shrinks when you move back
           left_cheek = lm[234]
           right_cheek = lm[454]
           face_width = math.hypot(right_cheek.x - left_cheek.x, right_cheek.y - left_cheek.y)


           top_face = lm[10]
           bottom_face = lm[152]
           face_height = math.hypot(top_face.y - bottom_face.y, top_face.x - bottom_face.x)


       if key == ord("c"):
           neutral_nose_x = nose_x
           neutral_nose_y = nose_y
           neutral_face_width = face_width
           neutral_head_height = face_height
           smooth_face_width = face_width
           smooth_face_height = face_height
           calibrated = True
           print("Calibrated")


       if calibrated:
           if smooth_face_width is None:
               smooth_face_width = face_width
           else:
               smooth_face_width = EMA_ALPHA * face_width + (1 - EMA_ALPHA) * smooth_face_width
           if smooth_face_height is None:
               smooth_face_height = face_height
           else:
               smooth_face_height = EMA_ALPHA * face_height + (1 - EMA_ALPHA) * smooth_face_height


           # Webcam preview is mirrored vs your body: use neutral_nose_x - nose_x so labels match physical turns.
           dx = neutral_nose_x - nose_x
           dy = nose_y - neutral_nose_y


           width_base = max(neutral_face_width, 1e-6)
           height_base = max(neutral_head_height, 1e-6)


           width_low = neutral_face_width - NEUTRAL_SIZE_TOLERANCE
           width_high = neutral_face_width + NEUTRAL_SIZE_TOLERANCE
           height_low = neutral_head_height - NEUTRAL_SIZE_TOLERANCE
           height_high = neutral_head_height + NEUTRAL_SIZE_TOLERANCE


           if width_low <= smooth_face_width <= width_high:
               width_reference = neutral_face_width
           elif smooth_face_width > width_high:
               width_reference = width_high
           else:
               width_reference = width_low


           if height_low <= smooth_face_height <= height_high:
               height_reference = neutral_head_height
           elif smooth_face_height > height_high:
               height_reference = height_high
           else:
               height_reference = height_low


           dw_norm = (smooth_face_width - width_reference) / width_base
           dh_norm = (smooth_face_height - height_reference) / height_base
           z_score = 0.6 * dw_norm + 0.4 * dh_norm


           sx = abs(dx) / max(X_THRESHOLD, 1e-9)
           sy = abs(dy) / max(Y_THRESHOLD, 1e-9)


           # Depth first, independent of head turn: forward/back never mix with left/right/up/down.
           if z_score > Z_ENTER_THRESHOLD:
               direction = "Moving forward"
           elif z_score < -Z_ENTER_THRESHOLD:
               direction = "Moving back"
           else:
               x_act = abs(dx) > X_THRESHOLD
               y_act = abs(dy) > Y_THRESHOLD
               best_planar = max(sx, sy)


               if not x_act and not y_act and best_planar < PLANAR_IDLE_MAX:
                   direction = "Looking straight"
               elif x_act and y_act:
                   if sx >= sy:
                       direction = "Looking right" if dx > 0 else "Looking left"
                   else:
                       direction = "Looking down" if dy > 0 else "Looking up"
               elif x_act:
                   direction = "Looking right" if dx > 0 else "Looking left"
               elif y_act:
                   direction = "Looking down" if dy > 0 else "Looking up"
               else:
                   direction = "Looking straight"
           print(direction)


           cv2.putText(image, direction, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
      
       else:
           cv2.putText(image, "Look straight and press C to calibrate", (30,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


   else:
       cv2.putText(image, "No head detected", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


   cv2.imshow('Face Tracking', image)


   key = cv2.waitKey(1) & 0XFF
   if key == ord("q"):
       break


video.release()
cv2.destroyAllWindows()