import tkinter as tk
from PIL import Image, ImageTk


import os
import sys

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def display_image(image_path):
    root = tk.Tk()
    root.title("Image Viewer")

    img = Image.open(image_path)
    photo = ImageTk.PhotoImage(img)

    label = tk.Label(root, image=photo)
    label.pack()

    root.mainloop()

if __name__ == "__main__":
    display_image(resource_path('planned_courses.png'))