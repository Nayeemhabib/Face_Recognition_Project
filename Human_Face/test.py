from sklearn.neighbors import KNeighborsClassifier

import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime

from win32com.client import Dispatch
def speak(strl):
    speak = Dispatch(("SAPI.SpVoice"))
    speak.speak(strl)


video=cv2.VideoCapture(0)
facedetect=cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

with open('data/names.pkl', 'rb') as f:
    LABELS=pickle.load(f)
with open('data/faces_data.pkl', 'rb') as f:
    FACES=pickle.load(f)

Knn = KNeighborsClassifier(n_neighbors=5)
Knn.fit(FACES,LABELS)

#imgBackground = cv2.imread("background.png")

col_names = ['Name', 'Date', 'Time']

while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray, 1.3 ,5)
    for (x,y,w,h) in faces:
        crop_img=frame[y:y+h, x:x+w, :]
        resized_img=cv2.resize(crop_img, (50,50)).flatten().reshape(1,-1)
        output = Knn.predict(resized_img)
        ts=time.time()
        date=datetime.fromtimestamp(ts).strftime("%d-%m-%y")
        timestamp=datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")

        cv2.rectangle(frame,(x,y), (x+w, y+h), (0,0,225),1)
        cv2.rectangle(frame,(x,y), (x+w, y+h), (50,50,225),2)
        cv2.rectangle(frame,(x,y-40), (x+w, y), (50,50,225),-1)

        attendence = [str(output[0]), str(timestamp)]
        cv2.putText(frame,str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255),1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
    #frame = imgBackground[162:162 + 480, 55:55 + 640]

    cv2.imshow("Frame",frame)
    k=cv2.waitKey(1)
    if k == ord('o'):
        speak("Attendance Taken")
        time.sleep(3)
        if exist:
            with open("Attendance/Attendance_" + date  + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendence)
            csvfile.close()
        else:
            with open("Attendance/Attendance_" + date  + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(col_names)
                writer.writerow(attendence)
            csvfile.close()
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()
