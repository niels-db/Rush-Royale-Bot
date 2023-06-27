import tkinter as tk
from tkinter import ttk
import shutil
import os

#You can change the file names to match your deck name if you wish.
def copy_file():
    if var.get() == "deck1":
        shutil.copy('./configs/deck1.ini', './config.ini')
    elif var.get() == "deck2":
        shutil.copy('./configs/deck2.ini', './config.ini')
    elif var.get() == "deck3":
        shutil.copy('./configs/deck3.ini', './config.ini')
    elif var.get() == "deck4":
        shutil.copy('./configs/deck4.ini', './config.ini')
    elif var.get() == "deck5":
        shutil.copy('./configs/deck5.ini', './config.ini')


root = tk.Tk()
root.title("Deck")
root.geometry("200x80")
# Set dark background
root.configure(background='#575559')
# Set window icon to png
root.iconbitmap('calculon.ico')
    
    
def move_window(event):
    root.geometry('+{0}+{1}'.format(event.x_root, event.y_root))

root.bind('<B1-Motion>', move_window)

var = tk.StringVar()

tk.Label(root, text="Select Deck").pack()

deck_options = ['Deck 1', 'Deck 2', 'Deck 3', 'Deck 4', 'Deck 5']
deck_combobox = ttk.Combobox(root, textvariable=var, values=deck_options)
deck_combobox.current(0) # Set the default option to the first one
deck_combobox.pack()

tk.Button(root, text="Submit", command=copy_file).pack()

