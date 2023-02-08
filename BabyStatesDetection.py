import datetime
import math

import cv2
import time
import numpy as np
import mediapipe as mp
import PoseModule as PM
import FaceMeshModule as FM
import FaceDetectionModule as FD

#cap = cv2.VideoCapture(0)     #开启相机捕获
cap = cv2.VideoCapture('PoseVideos/1.mp4')  # 读取视频

detector = PM.poseDetector()  # 检测并且画出身体图
detector2 = FM.FaceMeshDetector()  # 检测并且画出脸部
# detector3 = FD.FaceDetector()  # 脸部识别

pPer1=0
pPer2=0
pState=0 # 默认在睡觉

pTime = 0
tipIds = [15, 16]

while True:
    success, img = cap.read()
    #img3 = detector3.findFaces(img)     # 画脸

#    img = cv2.imread("PoseVideos/1.JPG")   # 读取读片

    img1 = detector.findPose(img,False)        #画身体
    img2, face = detector2.findFaceMesh(img,False)   # 画脸谱
    lmList = detector.findPosition(img1,False)  # 找到关键点的坐标

    bodyImg = detector.findBody(img1)
    detector2.drawEyes(img)

    if len(lmList)!=0:
        # 判断被子是否踢掉了
        angle1 = detector.findAngle(img, 24, 26, 28)  # 左腿弯曲角度
        angle2 = detector.findAngle(img, 23, 25, 27)  # 右腿弯曲角度
        per1 =np.interp(angle1, (120, 150), (0, 100))  # 将角度转化为百分制
        per2 =np.interp(angle1, (240, 210), (0, 100))  # 将角度转化为百分制
        # 求脚跟到大腿根部的距离
        legLength1 = math.sqrt((lmList[24][1]-lmList[28][1])**2+(lmList[24][2]-lmList[28][2])**2)
        leghold1 = 3*math.sqrt((lmList[12][1]-lmList[24][1])**2+(lmList[12][2]-lmList[24][2])**2)/5
        legLength2 = math.sqrt((lmList[24][1]-lmList[28][1])**2+(lmList[24][2]-lmList[28][2])**2)
        leghold2 = 3*math.sqrt((lmList[12][1]-lmList[24][1])**2+(lmList[12][2]-lmList[24][2])**2)/5
        # 左腿伸直检测
        if per1 !=100:
             pPer1 = 0
        if per1 ==100 and pPer1==0 and legLength1 >leghold1:
            print(state,",","kick leg",",",datetime.datetime.now().strftime('%Y-%m-%d %H:%H:%S'))
            cv2.putText(img1, str("kick leg"), (lmList[26][1]-10 , lmList[26][2]), cv2.FONT_HERSHEY_PLAIN,
                              3, (0, 255,0), 4)
            pPer1=1
        # 右脚伸直检测
        if per2 != 100:
             pPer2 = 0
        if per2 ==100 and pPer2 == 0 and legLength2 >leghold2:
            print(state,",","kick leg",",",datetime.datetime.now().strftime('%Y-%m-%d %H:%H:%S'))
            cv2.putText(img1, str("kick leg"), (lmList[25][1]+10, lmList[25][2]), cv2.FONT_HERSHEY_PLAIN,
                         3, (0, 255, 0), 4)
            pPer2 =1



        #检测是否醒了
        ix1= min(lmList[7][1], lmList[8][1])
        ix2= max(lmList[7][1], lmList[8][1])
        #判断睁眼的相关参数
        eyehold = math.sqrt((face[33][1]-face[155][1])**2+(face[33][2]-face[155][2])**2)  # 眼角距离
        leftEyeopen = math.sqrt((face[159][1]-face[144][1])**2+(face[159][2]-face[144][2])**2)
        rightEyeopen = math.sqrt((face[385][1]-face[374][1])**2+(face[385][2]-face[374][2])**2)


        if (lmList[15][1] > ix1 and lmList[15][1] < ix2) or (lmList[16][1] > ix1 and lmList[16][1] < ix2):
           #state = "awake"
            cv2.putText(img1, str("tuch face"), (lmList[7][1] , lmList[7][2]), cv2.FONT_HERSHEY_PLAIN,
                         2, (255, 0, 255), 2)
        # 眼睛睁开
        elif leftEyeopen > eyehold/3 or rightEyeopen > eyehold/3:
            state="awake"

        else:
              state = "sleep"
    if state=="awake":
        pState=0

    if state == "awake" and pState==0:
        print(state, ",", datetime.datetime.now().strftime('%Y-%m-%d %H:%H:%S'))
        pState=1

    cTime =time.time()
    fps =1/(cTime-pTime)
    pTime=cTime
    Mytime = datetime.datetime.now()

    cv2.putText(img1, str(state), (face[10][1:]), cv2.FONT_HERSHEY_PLAIN,
                3, (0, 255, 0), 3)  # 输出手指数

    cv2.putText(img, f'FPS:{int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN,
                2, (255, 0, 255), 2)            # PLAIN为字体
    cv2.putText(img, str(Mytime), (5, 30), cv2.FONT_HERSHEY_PLAIN,
                1, (0, 255, 0), 1)            # PLAIN为字体


    cv2.imshow("Image", img)
    cv2.waitKey(1)

















