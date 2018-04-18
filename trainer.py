import cv2
import os
import numpy as np
from PIL import Image # Python Image Library

recognizer = cv2.face.LBPHFaceRecognizer_create()
path = "Dataset"


# Function to determine the faces and corresponding ID number
def getImageWithID(path):
	imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
	faces = []
	IDs = []

	for imagePath in imagePaths:
		# Read image from the directory and convert numpy array
		faceImage = Image.open(imagePath).convert('L')
		faceNp = np.array(faceImage, 'uint8')

		ID = int(os.path.split(imagePath)[-1].split('.')[1])
		faces.append(faceNp)
		IDs.append(ID)

		cv2.imshow("Trainer", faceNp)
		cv2.waitKey(10)

	return IDs, faces


# Call the function with Dataset directory's path
ids, faces = getImageWithID(path)
recognizer.train(faces, np.array(ids))
recognizer.save("Recognizer/trainedData.yml")
cv2.destroyAllWindows()




