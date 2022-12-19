import cv2 as cv2
from tkinter import *
from tkinter import filedialog
from tkinter import font

filepath = ""
car_tracker_file = 'Trackers/car_detector.xml'
pedestrian_tracker_file = 'Trackers/pedestrian_detector.xml'


def openFile():
    global filepath
    filepath = filedialog.askopenfilename()
    print(filepath)
    video = cv2.VideoCapture(filepath)

    car_tracker = cv2.CascadeClassifier(car_tracker_file)
    pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

    while True:
        read_successful, frame = video.read()

        if read_successful:
            grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            break

        cars = car_tracker.detectMultiScale(grayscale_frame)
        pedestrians = pedestrian_tracker.detectMultiScale(grayscale_frame)

        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        for (x, y, w, h) in pedestrians:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

        cv2.imshow('Cars and pedestrians detector', frame)

        keyCode = cv2.waitKey(1)
        if cv2.getWindowProperty('Cars and pedestrians detector', cv2.WND_PROP_VISIBLE) < 1:
            break
        if keyCode == 27:
            cv2.destroyAllWindows()
            break


root = Tk(className="Group 9")

frame = Frame(root, bg="white", width=60, height=10)
frame.pack()

label_font = font.Font(family='Helvitica', size=20)
label = Label(frame, text="Cars and pedestrians detector",
              font=label_font, width=50, height=5, bg="white")
label.pack()

button_font = font.Font(family='Helvitica', size=14)
button = Button(frame, text="Open", command=openFile, bg='#45b592',
                fg='#ffffff',
                bd=0,
                font=button_font,
                height=1,
                width=6,
                pady=10)

button.pack()

root.mainloop()
