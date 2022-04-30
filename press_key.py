import keyboard
import os

def close_app():
    keyboard.press_and_release("alt+f4")

def save_doc():
    keyboard.press_and_release("ctrl+s")

def print_doc():
    keyboard.press_and_release("ctrl+p")

def restart_pc():
    restart = input("Do you want to restart your computer? ( y or n ) : ")
    if restart == "y" or restart == "Y":
        # 0 is time that is for after what time we want to restart
        os.system("shutdown /r /t 1")
    else:
        exit()

def shutdown_pc():
    shutdown = input("Do you want to shut down your computer? ( y or n ) : ")
    if shutdown == "y" or restart == "Y":
        os.system("shutdown /s /t 1")
    else:
        exit()

