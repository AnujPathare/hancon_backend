# Import required Libraries
from gzip import READ
from multiprocessing.spawn import prepare
from tkinter import *
import tkinter
from tokenize import String
from PIL import Image, ImageTk
import cv2
from cv2 import exp
import app_backend
# Create an instance of TKinter Window or frame
win = Tk()

#can1 = Canvas()
# Set the size of the window
win.geometry("1080x1920")

# Create a Label to capture the Video frames
label = Label(win)
label.grid(row=0, column=0)
cap = cv2.VideoCapture(0)

# Define function to show frame
svm = app_backend.prepare_model()


def show_frames():
    # Get the latest frame and convert into Image
    cv2image = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
    b1.destroy()
    predictions = app_backend.get_predictions(cv2image, svm)
    print(predictions)
    app_backend.map_to_keyboard(predictions)
    img = Image.fromarray(cv2image)
    # Convert image to PhotoImage
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    # Repeat after an interval to capture continiously
    label.after(20, show_frames)
    

b1 = Button(win, text="click me!",command=show_frames)
b1.grid(row=1,column=0,sticky=W,pady=2)
w = Label(win, text='Welcome to HANCON!',font="90",fg="Navyblue")
w.place(anchor=CENTER, relx=.1,rely=.1)
msg = Message(win,text="Click the button to start the application...")
msg.grid(row=1,column=1)
win.mainloop()
