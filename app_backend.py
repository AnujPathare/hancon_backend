import numpy as np
import cv2
import mediapipe as mp
import pandas as pd
import pickle
import time
import keyboard
import os


def mediapipe_detection(input_image, holistic):
    image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)
    return results


def extract_keypoints(results):
    hand_landmark = np.array([[res.x, res.y, res.z] for res in results.multi_hand_landmarks[0].landmark]).flatten(
    ) if results.multi_hand_landmarks[0] else np.zeros(21*3)
    return np.concatenate([hand_landmark])


columns = []
for i in range(63):
    columns.append(str(i))
label_map = {1: 'Close', 2: 'Print', 3: 'Restart', 0: 'Save'}

# key_map = {'Close': 'Alt + F4', 'Print': 'Ctrl + P',
#            'Restart': 'Alt + F4 + down'}


def prepare_model():
    time.sleep(10)
    file_name = 'C:/Anuj/Semesters/Sem6/1_MIni_Project/TKinter/hand_model_27_04.sav'
    hand_model = pickle.load(open(file_name, 'rb'))
    return hand_model


def get_predictions(frame, svm):
    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7) as hands:
        try:
            keypts = extract_keypoints(mediapipe_detection(frame, hands))
            predictions = str(label_map[svm.predict(pd.DataFrame([keypts], columns)).max()])
            print(predictions)
            return predictions
        except:
            pass
    


# def map_to_keyboard(predictions):
#     try:
#         key_map_value = key_map[predictions]
#         keyboard.press_and_release(key_map_value)
#     except:
#         pass

def map_to_keyboard(predictions):
    def close_app():
        keyboard.press_and_release("alt+f4")

    def save_doc():
        keyboard.press_and_release("ctrl+s")

    def print_doc():
        keyboard.press_and_release("ctrl+p")

    def restart_pc():
        # restart = input("Do you want to restart your computer? ( y or n ) : ")
        # if restart == "y" or restart == "Y":
            # 0 is time that is for after what time we want to restart
            os.system("shutdown /r /t 1")
        # else:
        #     exit()

    if (predictions == 'Close'):
        close_app()

    elif(predictions == 'Print'):
        print_doc()

    elif(predictions == 'Restart'):
        restart_pc()

    elif(predictions == 'Save'):
        save_doc()
        
    else:
        pass