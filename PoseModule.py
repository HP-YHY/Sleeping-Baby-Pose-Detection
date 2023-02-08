import math

import cv2
import mediapipe as mp
import time

# 创建一个工作类
class poseDetector():

    def __init__(self, mode=False, comPlex = 1, marks=True, segment=False, smooth=True,
                 detectionCon = 0.5, trackCon = 0.5):
# 参数
        self.mode = mode
        self.comPlex = comPlex
        self.marks = marks
        self.segment = segment
        self.smooth = smooth
        self.detectionCon = detectionCon  # 检测信任度
        self.trackCon = trackCon          # 追踪信任度

        self.mpDraw = mp.solutions.drawing_utils  # 声明
        self.mpPose = mp.solutions.pose  # 创建追踪对象
        self.pose = self.mpPose.Pose(self.mode, self.comPlex, self.marks, self.segment,self.smooth,
                                     self.detectionCon, self.trackCon)  # 传输参数
    def findPose(self, img, draw=True):   # 找到姿势，是否想要转换图像，是否想要检测姿势，是否要画
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换图像，这个图像是bgr,但引用的库和框架中使用的rgb
        self.results = self.pose.process(imgRGB)  # 检测姿势

        if self.results.pose_landmarks:
            if draw:
               self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                          self.mpPose.POSE_CONNECTIONS)  # 实时绘制点，连接点
        return img    # 返回图像

    def findPosition(self, img, draw = True):
        self.lmList = []   # 创建一个数组用于存放mark地址
        for id, lm in enumerate(self.results.pose_landmarks.landmark):
            h, w, c = img.shape
            #print(id, lm)   # 打印每个点的id
            cx, cy = int(lm.x*w), int(lm.y*h)  # 坐标强制转化为整数,
            self. lmList.append([id, cx, cy])      # 还可以存放Z轴，或者透明度
            if draw:
                cv2.circle(img, (cx, cy), 3, (0, 0, 255), cv2.FILLED) # 像素点绘制，大小为3，,绿色，实心
        return self.lmList
#找到三个特定的点
    def findAngle(self, img, p1, p2, p3, draw=True):
        x1, y1 = self.lmList[p1][1:]  # 取数组里面第二个和第三个
        x2, y2 = self.lmList[p2][1:]  # 取数组里面第二个和第三个
        x3, y3 = self.lmList[p3][1:]  # 取数组里面第二个和第三个
        # 估计三点连线的角度
        angle = math.degrees(math.atan2(y3-y2, x3-x2) -
                             math.atan2(y1-y2, x1-x2))
        # print(angle)
        # 画圆
        if draw:
            # 专门画连接这三个点的线
            cv2.line(img, (x1, y1),(x2, y2), (255, 0, 0), 2)
            cv2.line(img, (x3, y3),(x2, y2), (255, 0, 0), 2)
            # 特殊标注三个点
            cv2.circle(img, (x1, y1), 5, (255, 0, 0), cv2.FILLED)  # 图像，坐标，直径，颜色，填满/不填满
            cv2.circle(img, (x1, y1), 10, (255, 0, 0), 1)  # 默认不填满，厚度为2

            cv2.circle(img, (x2, y2), 5, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 0), 1)

            cv2.circle(img, (x3, y3), 5, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x3, y3), 10, (255, 0, 0), 1)
            cv2.putText(img, str(int(angle)),(x2 - 10, y2 + 20),
                        cv2.FONT_HERSHEY_PLAIN,1,(255, 0, 255), 2)
        return angle

    def findBody(self,img,draw=True):
        self.green = [0,255,0]
        self.red = [0,0,255]
        self.blue = [255,0,0]
        self.color =self.blue
        if draw:
            cv2.line(img, (self.lmList[11][1:]), (self.lmList[12][1:]), self.color, 2)
            cv2.line(img, (self.lmList[11][1:]), (self.lmList[13][1:]), self.color, 2)
            cv2.line(img, (self.lmList[11][1:]), (self.lmList[23][1:]), self.color, 2)
            cv2.line(img, (self.lmList[12][1:]), (self.lmList[14][1:]), self.color, 2)
            cv2.line(img, (self.lmList[12][1:]), (self.lmList[24][1:]), self.color, 2)
            cv2.line(img, (self.lmList[13][1:]), (self.lmList[15][1:]), self.color, 2)
            cv2.line(img, (self.lmList[14][1:]), (self.lmList[16][1:]), self.color, 2)
            cv2.line(img, (self.lmList[23][1:]), (self.lmList[24][1:]), self.color, 2)
            cv2.line(img, (self.lmList[23][1:]), (self.lmList[25][1:]), self.color, 2)
            cv2.line(img, (self.lmList[24][1:]), (self.lmList[26][1:]), self.color, 2)
            cv2.line(img, (self.lmList[25][1:]), (self.lmList[27][1:]), self.color, 2)
            cv2.line(img, (self.lmList[26][1:]), (self.lmList[28][1:]), self.color, 2)


            cv2.circle(img, (self.lmList[16][1:]), 5, (255, 0, 0), cv2.FILLED)  # 图像，坐标，直径，颜色，填满/不填满
            cv2.circle(img, (self.lmList[16][1:]), 10, (255, 0, 0), 1)  # 默认不填满，厚度为2

            cv2.circle(img, (self.lmList[15][1:]), 5, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (self.lmList[15][1:]), 10, (255, 0, 0), 1)
            cv2.circle(img, (self.lmList[11][1:]), 5, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (self.lmList[11][1:]), 10, (255, 0, 0), 1)
            cv2.circle(img, (self.lmList[12][1:]), 5, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (self.lmList[12][1:]), 10, (255, 0, 0), 1)

# 检测帧率
#放入主函数的部分
def main():
    cap = cv2.VideoCapture('PoseVideos/1.mp4')  # 读取视频
    pTime = 0  # 前一时间
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)

        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[14])   # 打印列表,只打印第14个坐标
            #特意加粗指定点，换成红色
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 5, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (lmList[13][1], lmList[13][2]), 5, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (lmList[28][1], lmList[28][2]), 5, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (lmList[27][1], lmList[27][2]), 5, (255, 0, 0), cv2.FILLED)

        cTime = time.time()
        fps = 1/(cTime-pTime)  # 帧率
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Image", img)

        cv2.waitKey(10)              # 一毫秒延迟

    # 单独运行此文件后，会跳转至main函数，运行其他函数后不会运行此函数
if __name__ == "__main__":
    main()
