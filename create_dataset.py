import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

id = input("Enter user ID number: ")
sampleNumber = 0

while True:
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	for (x_coor, y_coor, width, height) in faces:
		sampleNumber = sampleNumber + 1
		cv2.imwrite("Dataset/User." + str(id) + "." + str(sampleNumber) + ".jpg", gray[y_coor:y_coor+height,x_coor:x_coor+width])
		cv2.rectangle(img, (x_coor,y_coor), (x_coor+width,y_coor+height), (0,0,255), 2)
		cv2.waitKey(100)

	cv2.imshow("Faces", img)
	cv2.waitKey(1)
	if sampleNumber > 30:
		break

cap.release()
cv2.destroyAllWindows()

