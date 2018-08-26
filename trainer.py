import os
import cv2
import numpy as np
from PIL import Image
rec = cv2.face.LBPHFaceRecognizer_create()
path='dataSet'
def getImageswithID(path):
	imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
	faces = []
	IDs = []

	for imgpath in imagePaths:
		faceImg = Image.open(imgpath)
		faceNp = np.array(faceImg,'uint8')
		ID = int(os.path.split(imgpath)[-1].split('_')[0])
		faces.append(faceNp)
		IDs.append(ID)
		cv2.imshow('training',faceNp)
		cv2.waitKey(10)
	return IDs, faces

IDs,faces = getImageswithID(path)
rec.train(faces,np.array(IDs))
rec.write('trainingData.yml')
cv2.destroyAllWindows()



