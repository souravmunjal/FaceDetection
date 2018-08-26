import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainingData.yml')
cascadePath = "haarcascade_frontalface_default.xml"
facedetect = cv2.CascadeClassifier(cascadePath);

id = 0 
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (0,0,255)

id_map = ['Sourav','Tarun']

cam = cv2.VideoCapture(0)

while(True):
    ret,img = cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray,(5,5))
    faces=facedetect.detectMultiScale(gray,1.3,5)

    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        id,conf = recognizer.predict(gray[y:y+h,x:x+w])

        cv2.putText(img,str(id_map[id-1])+'_'+str(conf),(x,y+h),fontFace,fontScale,fontColor)

    cv2.imshow("face",img)
    if(cv2.waitKey(1)==ord('q')):
        break

cam.release()
cv2.destroyAllWindows()
