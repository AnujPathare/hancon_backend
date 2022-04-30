from tkinter import *
from PIL import ImageTk, Image
import cv2
import pickle
import pandas as pd
import numpy as np
import mediapipe as mp
# import face_recognition
# import keyboard
from time import time

win = Tk()
win.grid()
# Create a label in the frame
lmain = Label(win)
lmain.grid()

# Capture from camera
cap = cv2.VideoCapture(0)

pkl_filename = "hand_model_27_04.sav"
with open(pkl_filename, 'rb') as file:
   hand_model = pickle.load(file)

columns = []
for i in range(63):
  columns.append(str(i))
label_dict = {0:'Close', 2:'Print', 3: 'Restart', 1: 'Save'}


# IMAGE_FILES = []
# mp_hands = mp.solutions.hands
# with mp_hands.Hands(
#     static_image_mode=True,
#     max_num_hands=2,
#     min_detection_confidence=0.5) as hands:
#   for idx, file in enumerate(IMAGE_FILES):
#     # Read an image, flip it around y-axis for correct handedness output (see
#     # above).
#     image = cv2.flip(cv2.imread(file), 1)
#     # Convert the BGR image to RGB before processing.
#     results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def mediapipe_detection(input_image, holistic):
    image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)
    return results


def extract_keypoints(results):
    hand_landmark = np.array([[res.x, res.y, res.z] for res in results.multi_hand_landmarks[0].landmark]).flatten(
    ) if results.multi_hand_landmarks[0] else np.zeros(21*3)
    return np.concatenate([hand_landmark])

# def recognize_me():
#   frame = frame
#   my_face_encoding = face_recognition.face_encodings(frame)[0]
#   return my_face_encoding

# def crop_image(img,results):
#     try:
#         h, w, c = img.shape
#         cx_min=  w
#         cy_min = h
#         cx_max= cy_max= 0
#         for id, lm in enumerate(results.face_landmarks.landmark):
#             cx, cy = int(lm.x * w), int(lm.y * h)
#             if cx<cx_min:
#                 cx_min=cx
#             if cy<cy_min:
#                 cy_min=cy
#             if cx>cx_max:
#                 cx_max=cx
#             if cy>cy_max:
#                 cy_max=cy
#         return img[cy_min - 10 :cy_max + 10, cx_min - 10:cx_max + 10]
#     except:
#         pass

# def identify_is_it_me(cropped_img):
#   try:
#     unknown_picture = cropped_img
#     unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]
#     results = face_recognition.compare_faces([recognize_me()], unknown_face_encoding)
#     return results[0]
#   except:
#     return False

with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
  vid = cv2.VideoCapture(0)
  while(True):
    current_gesture = ''
    ret, frame = vid.read()
    results = mediapipe_detection(frame, holistic)
    cv2.imshow("gesfrme", frame)
    # if identify_is_it_me(crop_image(frame,results)):

    key = extract_keypoints(mediapipe_detection(frame, holistic))
    key_data = pd.DataFrame([key],columns = columns)
    prediction = hand_model.predict(key_data)
    current_gesture =  label_dict[int(prediction)]
    if list(hand_model.predict_proba(key_data)[0])[int(prediction)] >= 0.8:
      print("Recognized gesture is: " + current_gesture)
      time.sleep(0)
    current_gesture = ''
  # else:
  #   pass
  #   #print("You are not authorized!")
  
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  # # After the loop release the cap object
  # vid.release()
  # # Destroy all the windows
  # cv2.destroyAllWindows()

# function for video streaming
def webcam_video():
    _, frame = cap.read()
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(20, webcam_video)

    # keypts = extract_keypoints(mediapipe_detection(frame, hands))
    # label_map[svm.predict(pd.DataFrame([keypts], columns)).max()]

webcam_video()
win.mainloop()

