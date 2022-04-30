# Import required Libraries
from tkinter import *
from PIL import Image, ImageTk
import cv2
import pickle
import pandas as pd
import numpy as np
import subprocess as sub

# Create an instance of TKinter Window or frame
win = Tk()

# Set the size of the window
win.geometry("650x500")

# Create a Label to capture the Video frames
label =Label(win)
label.grid(row=0, column=0)
cap= cv2.VideoCapture(0)


# x_tf = Entry(win)
# x_tf.grid(row=750, column=600)


# pkl_filename = "hand_model_27_04.sav"
# with open(pkl_filename, 'rb') as file:
#    pickle_model = pickle.load(file)

# columns = []
# for i in range(63):
#   columns.append(str(i))
# label_dict = {0:'Close', 2:'Print', 3: 'Restart', 1: 'Save'}

# def mediapipe_detection(input_image, holistic):
#     image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
#     results = holistic.process(image)
#     return results

# def extract_keypoints(results):
#     hand_landmark = np.array([[res.x, res.y, res.z] for res in results.multi_hand_landmarks[0].landmark]).flatten(
#     ) if results.multi_hand_landmarks[0] else np.zeros(21*3)
#     return np.concatenate([hand_landmark])

# Define function to show frame
def show_frames():
   # Get the latest frame and convert into Image
   cv2image= cv2.cvtColor(cap.read()[1],cv2.COLOR_BGR2RGB)
   img = Image.fromarray(cv2image)
   # Convert image to PhotoImage
   imgtk = ImageTk.PhotoImage(image = img)
   label.imgtk = imgtk
   label.configure(image=imgtk)
   # Repeat after an interval to capture continiously
   label.after(20, show_frames)

  #  keypts = extract_keypoints(mediapipe_detection(frame, hands))
  #  label_map[svm.predict(pd.DataFrame([keypts], columns)).max()]
   
'''colourImg = img
   colourPixels = colourImg.convert("RGB")
   colourArray = np.array(colourPixels.getdata()).reshape(colourImg.size + (3,))
   indicesArray = np.moveaxis(np.indices(colourImg.size), 0, 2)
   allArray = np.dstack((indicesArray, colourArray)).reshape((-1, 63)) 

   df = pd.DataFrame(allArray, columns=["y", "x", "red","green","blue"])

   pickle_model.predict(df) '''

show_frames()
win.mainloop()