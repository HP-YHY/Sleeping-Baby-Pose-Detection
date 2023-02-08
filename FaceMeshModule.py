
import cv2
import mediapipe as mp
import time
import math

class FaceMeshDetector():

    def __init__(self, mode = False, maxFaces = 2, landmarks = False, detectionConfi = 0.5, trackingConfi = 0.5):

        self.mode = mode
        self.maxFaces = maxFaces
        self.landmarks = landmarks
        self.detectionConfi = detectionConfi
        self.trackingConfi = trackingConfi
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.mode, self.maxFaces, self.landmarks,
                                                 self.detectionConfi, self.trackingConfi)  # 把其中一个默认的值改掉
        self.drawSpec1 = self.mpDraw.DrawingSpec(color= (0, 255, 0), circle_radius= 1)    # 修改默认参数，红色，线宽，直径，
        self.drawSpec2 = self.mpDraw.DrawingSpec(color= (0, 255, 0), thickness=1)    # 修改默认参数，绿色，线宽，直径，

    def findFaceMesh(self,img,draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)  # 处理脸谱，画脸谱
        self.faces = []
        if self.results.multi_face_landmarks:

            for self.faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, self.faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec1
                                          , self.drawSpec2)  # 边框线 标志点
                    # 输出标志点坐标
                self.face=[]
                for id, lm in enumerate(self.faceLms.landmark):
                    # print(lm)
                    ih, iw, ic = img.shape
                    x, y = int(lm.x*iw), int(lm.y*ih)  # 标准化坐标

                    # cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN,
                    #            1, (0, 255, 0), 1)    # 每个点变成了id

                    # print(id, x, y)
                    self.face.append([id, x, y]) # 将x y 存储进face
                    #print(self.face)
                self.faces.append(self.face) # 将face 存储进faces
        return img, self.face

    def drawEyes(self, img, draw=True):
       # print(img, self.face[16])
        # 眼睛颜色
        self.color1 = [0,125,0]
        # 脸边框颜色
        self.color2 = [255, 0, 0]
        self.thinckness = 2
        cv2.line(img,(self.face[33][1:]), (self.face[160][1:]), self.color1, self.thinckness)
        cv2.line(img,(self.face[160][1:]), (self.face[159][1:]), self.color1, self.thinckness)
        cv2.line(img,(self.face[159][1:]), (self.face[158][1:]), self.color1, self.thinckness)
        cv2.line(img,(self.face[158][1:]), (self.face[155][1:]), self.color1, self.thinckness)
        cv2.line(img,(self.face[155][1:]), (self.face[145][1:]), self.color1, self.thinckness)
        cv2.line(img,(self.face[145][1:]), (self.face[144][1:]), self.color1, self.thinckness)
        cv2.line(img,(self.face[144][1:]), (self.face[163][1:]), self.color1, self.thinckness)
        cv2.line(img,(self.face[163][1:]), (self.face[33][1:]), self.color1, self.thinckness)

        cv2.line(img,(self.face[463][1:]), (self.face[384][1:]), self.color1, self.thinckness)
        cv2.line(img,(self.face[384][1:]), (self.face[385][1:]), self.color1, self.thinckness)
        cv2.line(img,(self.face[385][1:]), (self.face[386][1:]), self.color1, self.thinckness)
        cv2.line(img,(self.face[386][1:]), (self.face[263][1:]), self.color1, self.thinckness)
        cv2.line(img,(self.face[263][1:]), (self.face[373][1:]), self.color1, self.thinckness)
        cv2.line(img,(self.face[373][1:]), (self.face[380][1:]), self.color1, self.thinckness)
        cv2.line(img,(self.face[380][1:]), (self.face[463][1:]), self.color1, self.thinckness)

        self.cirled=1
        # 左眼
        cv2.circle(img, (self.face[33][1:]), self.cirled, self.color1, cv2.FILLED)
        cv2.circle(img, (self.face[160][1:]), self.cirled, self.color1, cv2.FILLED)
        cv2.circle(img, (self.face[159][1:]),self.cirled, self.color1, cv2.FILLED)
        cv2.circle(img, (self.face[158][1:]),self.cirled, self.color1, cv2.FILLED)
        cv2.circle(img, (self.face[155][1:]), self.cirled, self.color1, cv2.FILLED)
        cv2.circle(img, (self.face[145][1:]), self.cirled,self.color1, cv2.FILLED)
        cv2.circle(img, (self.face[144][1:]), self.cirled, self.color1, cv2.FILLED)
        cv2.circle(img, (self.face[163][1:]), self.cirled, self.color1, cv2.FILLED)
        # 右眼
        cv2.circle(img, (self.face[463][1:]), self.cirled, self.color1, cv2.FILLED)
        cv2.circle(img, (self.face[384][1:]), self.cirled, self.color1, cv2.FILLED)
        cv2.circle(img, (self.face[385][1:]), self.cirled,self.color1, cv2.FILLED)
        cv2.circle(img, (self.face[386][1:]), self.cirled, self.color1, cv2.FILLED)
        cv2.circle(img, (self.face[263][1:]), self.cirled, self.color1, cv2.FILLED)
        cv2.circle(img, (self.face[373][1:]), self.cirled, self.color1, cv2.FILLED)
        cv2.circle(img, (self.face[374][1:]), self.cirled, self.color1, cv2.FILLED)
        cv2.circle(img, (self.face[380][1:]), self.cirled, self.color1, cv2.FILLED)

        # 画一个脸部方框
        faceCirle = math.sqrt((self.face[6][1]-self.face[58][1])**2
                              +(self.face[6][2]-self.face[58][2])**2)
        cv2.circle(img, (self.face[6][1:]), int(faceCirle), self.color2, 2)

        #画脸







def main():
    # cap = cv2.VideoCapture("video/2.mp4")
    pTime = 0
    detector = FaceMeshDetector()
    while True:
        # success, img = cap.read()

        img = cv2.imread("video/2.JPG")  # 读取读片
        img = cv2.resize(img, (1200, 900))
        img, faces = detector.findFaceMesh(img, False)  # 画点


        detector.drawEyes(img)
        # if len(faces)!= 0:
        #      print(faces[12])

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 255, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()








