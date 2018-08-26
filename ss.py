import numpy as np 
import cv2


facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)


Id=input('Enter user id')
i=0;
while True:
    check,frame=cam.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray,(5,5))
    faces=facedetect.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        i=i+1
        cv2.imwrite('dataSet/'+str(Id)+'_'+str(i)+'.jpg',gray[y:y+h,x:x+w])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.waitKey(100)
    print(check)
    print(frame)
    cv2.imshow("ss",frame)
    key=cv2.waitKey(1)
    if i>20:
        break;

cam.release()
cv2.destroyAllWindows()
