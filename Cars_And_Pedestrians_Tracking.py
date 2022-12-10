import cv2 as cv2
from tkinter import *
from tkinter import filedialog

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

window = Tk()
button = Button(text="Open",command=openFile)
button.pack()

window.mainloop()


