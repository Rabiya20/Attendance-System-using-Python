from datetime import datetime
import face_recognition
import numpy as np
import cv2
import os

imgPath = 'images'
images = []
personName = []
myList = os.listdir(imgPath)
print(myList)

for cv_img in myList:
    current_image = cv2.imread(f'{imgPath}/{cv_img}')
    images.append(current_image)
    personName.append(os.imgPath.splitext(cv_img)[0])

print(personName)

def faceEncoding(images):
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

print(faceEncoding[images])

encodeListKnown = faceEncoding(images)
print("All Encodings Completed!")


def attendance(name):
    with open('attendance-marked.csv', 'r+') as f:
        dataList = f.readlines()
        nameList = []
        for line in dataList:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            timeNow = datetime.now()
            timeStr = time_now.strtime('%H:%M:%S')
            dateStr = time_now.strtime('%d/%m/%y')
            f.writelines(f'{name}, {timeStr}, {dateStr}')


capture = cv2.VideoCapture(1)
while True:
    ret, frame = capture.read()
    faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    currentFaceFrame = face_recognition.face_locations(faces)
    encodeCurrentFrame = face_recognition.face_encodings(faces, currentFaceFrame)

    for encodeFace, faceLoc in zip(encodeCurrentFrame, currentFaceFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = personName[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            attendance(name)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(10) == 13:
        break

capture.release()
cv2.destroyAllWindows()

