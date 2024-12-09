import tkinter as tk
from tkinter import filedialog, ttk
from PIL import ImageTk, Image
import numpy as np
from keras.models import load_model # type: ignore
import time

#model
model = load_model('traffic_classifier.h5')

#dictionary
classes = {
    1: 'Speed limit (20km/h)',
    2: 'Speed limit (30km/h)',
    3: 'Speed limit (50km/h)',
    4: 'Speed limit (60km/h)',
    5: 'Speed limit (70km/h)',
    6: 'Speed limit (80km/h)',
    7: 'End of speed limit (80km/h)',
    8: 'Speed limit (100km/h)',
    9: 'Speed limit (120km/h)',
    10: 'No passing',
    11: 'No passing veh over 3.5 tons',
    12: 'Right-of-way at intersection',
    13: 'Priority road',
    14: 'Yield',
    15: 'Stop',
    16: 'No vehicles',
    17: 'Veh > 3.5 tons prohibited',
    18: 'No entry',
    19: 'General caution',
    20: 'Dangerous curve left',
    21: 'Dangerous curve right',
    22: 'Double curve',
    23: 'Bumpy road',
    24: 'Slippery road',
    25: 'Road narrows on the right',
    26: 'Road work',
    27: 'Traffic signals', 
    28: 'Pedestrians',
    29: 'Children crossing',
    30: 'Bicycles crossing',
    31: 'Beware of ice/snow',
    32: 'Wild animals crossing',
    33: 'End speed + passing limits',
    34: 'Turn right ahead',
    35: 'Turn left ahead',
    36: 'Ahead only',
    37: 'Go straight or right',
    38: 'Go straight or left',
    39: 'Keep right',
    40: 'Keep left',
    41: 'Roundabout mandatory',
    42: 'End of no passing',
    43: 'End of no passing vehicle with a weight greater than 3.5 tons'
}

#app
top = tk.Tk()
top.geometry('800x600')
top.title('Traffic Sign Detection And Classification')
top.configure(background='#eef2f3')

font_heading = ('Poppins', 20, 'bold')
font_label = ('Poppins', 14)

label = tk.Label(top, background='#eef2f3', font=font_label)
sign_image = tk.Label(top)

progress = ttk.Progressbar(top, orient='horizontal', length=300, mode='indeterminate')

#classify animaton
def start_animation():
    progress.pack(pady=10)
    progress.start()

def stop_animation():
    progress.stop()
    progress.pack_forget()

def classify(file_path):
    start_animation()
    label.configure(text='Classifying...', foreground='#888888', font=('Poppins', 14, 'italic'))
    top.update()
    time.sleep(1)
    stop_animation()

    try:
        image = Image.open(file_path).resize((30, 30))
        image = np.expand_dims(np.array(image), axis=0)
        prediction = model.predict(image) #prediction
        pred = np.argmax(prediction, axis=-1)[0]
        sign = classes.get(pred + 1, "Unknown Sign")

        label.configure(
            text=f"Prediction: {sign}",
            foreground='#ffffff', 
            background='#007bff', 
            font=('Poppins', 16, 'bold'),
            padx=20,
            pady=10,
            relief="solid",
            bd=2,  
            width=30,
            anchor='center'
        )
    except Exception as e:
        label.configure(foreground='red', text=f"Error: {e}", font=('Poppins', 14, 'italic')) #error sentence if there is any


def show_classify_button(file_path):
    classify_b = tk.Button(
        top, text="Classify Image", command=lambda: classify(file_path), padx=10, pady=5
    )
    classify_b.configure(background='#007bff', foreground='white', font=font_label, borderwidth=0)
    classify_b.place(relx=0.79, rely=0.46)

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        uploaded = Image.open(file_path).resize((250, 250))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)

upload = tk.Button(
    top, text="Upload an Image", command=upload_image, padx=10, pady=5
)
upload.configure(background='#28a745', foreground='white', font=font_label, borderwidth=0)
upload.pack(side=tk.BOTTOM, pady=50)


#icon
icon_path = 'icon.ico'
app_icon = ImageTk.PhotoImage(Image.open(icon_path).resize((50, 50)))
top.iconphoto(False, app_icon)

#image
icon_image = ImageTk.PhotoImage(Image.open('icon.png').resize((100, 100)))
icon_label = tk.Label(top, image=icon_image, background='#eef2f3')
icon_label.pack(pady=10)

sign_image.pack(side=tk.BOTTOM, expand=True)
label.pack(side=tk.BOTTOM, expand=True)

heading = tk.Label(top, text="Traffic Sign Detection And Classification", pady=20, font=font_heading)
heading.configure(background='#eef2f3', foreground='#333333')
heading.pack()

top.mainloop()