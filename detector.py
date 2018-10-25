import cv2
import numpy as np
from pynput.mouse import Button,Controller
import wx
import time

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainingData.yml')
cascadePath = "haarcascade_frontalface_default.xml"
facedetect = cv2.CascadeClassifier(cascadePath);

id = 0 
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (0,0,255)

id_map = ['Sourav','Tarun','Paras']

cam = cv2.VideoCapture(0)

def my_function():
    mouse=Controller()
    app=wx.App(False)
    (sx,sy)=wx.GetDisplaySize()
    (camx,camy)=(640,480)


    lowerBound=np.array([33,80,40])
    upperBound=np.array([102,255,255])

    cam= cv2.VideoCapture(0)
    cam.set(3,camx)
    cam.set(4,camy)
    kernelOpen=np.ones((5,5))
    kernelClose=np.ones((20,20))

    mLocOld=np.array([0,0])
    mouseLoc=np.array([0,0])
    DampingFactor=2 # should be more than 1 
    #mouseLoc=mLocOld+(targetLoc-mLocOld)/DampingFactor

    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (0,255,255)
    openx,openy,openw,openh=(0,0,0,0)
    pinchFlag=0

    while True:
        ret, img=cam.read()
        img=cv2.resize(img,(640,480))

        #convert BGR to HSV
        imgHSV= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        # create the Mask
        mask=cv2.inRange(imgHSV,lowerBound,upperBound)
        #morphology
        maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
        maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)

        maskFinal=maskClose
        _, conts, _=cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        if(len(conts)==2):
            if(pinchFlag==1):
                pinchFlag=0
                mouse.release(Button.left)
            x1,y1,w1,h1=cv2.boundingRect(conts[0])
            x2,y2,w2,h2=cv2.boundingRect(conts[1])
            cv2.rectangle(img,(x1,y1),(x1+w1,y1+h1),(255,0,0),2)
            cv2.rectangle(img,(x2,y2),(x2+w2,y2+h2),(255,0,0),2)
            cx1=int(x1+w1/2)
            cy1=int(y1+h1/2)
            cx2=int(x2+w2/2)
            cy2=int(y2+h2/2)
            cx=int((cx1+cx2)/2)
            cy=int((cy2+cy1)/2)
            cv2.line(img,(cx1,cy1),(cx2,cy2),(255,0,0,),2)
            cv2.circle(img,(cx,cy),2,(0,0,255),2)
            mouseLoc=mLocOld+((cx,cy)-mLocOld)/DampingFactor
            mouse.position=(sx-(mouseLoc[0]*sx/camx),mouseLoc[1]*sy/camy)
            mLocOld=mouseLoc
            openx,openy,openw,openh=cv2.boundingRect(np.array([[[x1,y1],[x1+w1,y1+h1],[x2,y2],[x2+w2,y2+w2]]]))                        
            cv2.rectangle(img,(openx,openy),(openx+openw,openy+openh),(255,0,0),2)
            
        elif(len(conts)==1):
            x,y,w,h=cv2.boundingRect(conts[0])
            if(pinchFlag==0):
                if (abs(w*h-openw*openh)*100/(w*h))<20:
                    pinchFlag=1
                    mouse.press(Button.left)

            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cx=int(x+w/2)
            cy=int(y+h/2)
            cv2.circle(img,(cx,cy),int((w+h)/4),(0,0,255),2)
            mouseLoc=mLocOld+((cx,cy)-mLocOld)/DampingFactor
            mouse.position=(sx-(mouseLoc[0]*sx/camx),mouseLoc[1]*sy/camy)
            mLocOld=mouseLoc
             
        
        #cv2.imshow("maskClose",maskClose)
        #cv2.imshow("maskOpen",maskOpen)
        #cv2.imshow("mask",mask)
        cv2.imshow("cam",img)
        cv2.waitKey(5)


while(True):
    ret,img = cam.read()
    img=cv2.resize(img,(640,480))
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray,(5,5))
    faces=facedetect.detectMultiScale(gray,1.3,5)
    
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        id,conf = recognizer.predict(gray[y:y+h,x:x+w])
        if(id!=0):
            print("Your identity is Confirmed")
            cv2.putText(img,str(id_map[id-1]),(int(x),int(y+h/2)),fontFace,fontScale,fontColor)
            cv2.putText(img,"Your Identity is Confirmed",(int(x),int(y+h/2)+20),fontFace,fontScale,fontColor)
            my_function()
            
            
    cv2.imshow("face",img)
    if(cv2.waitKey(1)==ord('q')):
        break
    

cv2.destroyAllWindows()
