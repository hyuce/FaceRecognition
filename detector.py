import cv2
import numpy as np
import datetime

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create() # create recognizer object
rec.read("Recognizer/trainedData.yml") # read the trained data

id = 0
font = cv2.FONT_HERSHEY_SIMPLEX

f = open("log.txt", 'a') # open a file to log time and date info

while True:
	ret, img = cap.read() # capture a frame from the cam
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(img, 1.3, 5) # detect faces in captured frame

	for (x_coor,y_coor,width,height) in faces:
		cv2.rectangle(img, (x_coor,y_coor), (x_coor+width,y_coor+height), (0,0,255), 2)
		pre = rec.predict(gray[y_coor:y_coor+height,x_coor:x_coor+width])

		id = pre[0]
		now = datetime.datetime.today()

		f.write(str(id)+" "+str(now)+"\n")

		cv2.putText(img, str(id), (x_coor+10,y_coor+height-10), font, 0.5, (0,0,255), 2, cv2.LINE_AA)

	cv2.imshow("Face", img)
	k = cv2.waitKey(1) & 0xFF
	if k == 27:
		break

cv2.imwrite("Screenshots/Image"+str(id)+".jpg", img)
f.close()
cap.release()
cv2.destroyAllWindows()
