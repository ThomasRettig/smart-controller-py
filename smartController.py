import sys
from PyQt5 import QtGui
from PyQt5.QtWidgets import (QWidget, QPushButton, QStyle, QSlider, QLabel, QApplication,
                             QMainWindow, QHBoxLayout, QVBoxLayout, QFrame)
from PyQt5.QtCore import QThread, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
import os, time
import vlc
try:
    import Queue as Queue
except:
    import queue as Queue

# Delay between every image capture through camera
# 相机每次拍摄之间的延迟
DISP_MSEC   = 0.1
# Memory queue to hold camera captured images
# 内存队列以保存相机拍摄的图像
image_queue = Queue.Queue()

class CameraCaptureThread(QThread):
    # create the PyQt signal to video window
    # 创建PyQt信号到视频窗口
    frontalFaceDetection = pyqtSignal(bool)

    def __init__(self, faceDetectionThresholdInSec, imageQ):
        super(CameraCaptureThread, self).__init__()
        # flag of the frontal face detection state
        # 正面检测状态的标志
        self.frontalFaceDetected = False
        # image queue to buffer the image frame
        # 图像队列以缓冲图像帧
        self.imageQ = imageQ
        # time in seconds when there is a facial detection state change
        # 面部检测状态改变时的时间（以秒为单位）
        self.frontalFaceDetectionStateChangeTime = time.perf_counter()
        # threshold in second to emit signal of detecting or losing a frontal face
        # 阈值，以秒为单位发出检测到或丢失正面的信号
        self.faceDetectionThresholdInSec = faceDetectionThresholdInSec
        # the detected image need to have a confidence level of at least 105
        # 检测到的图像必须具有至少105的置信度
        self.decisionThreshold = 105.0
        # initialize the cascade classifier with the haarcascade_frontalface_alt.xml model
        # 使用haarcascade_frontalface_alt.xml模型初始化cascade classifier
        self.upper_body_cascade = cv2.CascadeClassifier(os.getcwd() + '/haarcascade_frontalface_alt.xml')

    def run(self):
        vCapture = cv2.VideoCapture(0)
        # never ending loop
        # 永无止境的循环
        while True:
            # capture image from camera
            # 从相机捕获图像
            ret, frame = vCapture.read()
            # when valid image is captured
            # 捕获有效图像时
            if ret:
                # transform color image to gray image
                # 将彩色图像转换为灰色图像
                grayImage = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

                # detect frontal human face in the image with opencv cascade classifier
                # 使用opencv的cascade classifier检测图像中的人脸正面
                # will return the bounding boxes for the detected objects
                # 将返回检测到的物件的边界框
                detected = self.upper_body_cascade.detectMultiScale3(grayImage, 1.1, outputRejectLevels=True)

                # eliminate detected human face if its confident level is below the required
                # 如果其置信度低于要求，则消除检测到的人脸
                boxes = np.array(detected[0])
                weights = np.array(detected[2]).flatten()
                # debugging outputs
                # print('boxes=',boxes)
                # print('weights=', weights)
                if len(weights) > 0:
                    boxes = boxes[weights > self.decisionThreshold]

                # box the detected object in green frame
                # 将检测到的物件放入绿色框
                boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
                for (xA, yA, xB, yB) in boxes:
                    cv2.rectangle(frame, (xA, yA), (xB, yB),
                                  (0, 255, 0), 2)

                # checking the human face detection state - detected or lose detection
                # 检查人脸检测状态-检测到或丢失检测
                # faces detected
                # 检测到人脸
                if len(boxes) > 0:
                    # update the face detection state and state change time
                    # 更新面部检测状态和状态的更改时间
                    if self.frontalFaceDetected == False:
                        self.frontalFaceDetected = True
                        self.frontalFaceDetectionStateChangeTime = time.perf_counter()
                # no face detected
                # 没检测到人脸
                else:
                    # update the face detection state and state change time
                    # 更新面部检测状态和状态的更改时间
                    if self.frontalFaceDetected == True:
                        self.frontalFaceDetected = False
                        self.frontalFaceDetectionStateChangeTime = time.perf_counter()

                # check if the detection state change exceed the required timing threshold
                # 检查检测状态变化是否超过所需的定时阈值
                if time.perf_counter() - self.frontalFaceDetectionStateChangeTime > self.faceDetectionThresholdInSec:
                    # emit the signal to video window on the detection state
                    # 在检测状态发射到视频窗口
                    if self.frontalFaceDetected == True:
                        self.frontalFaceDetection.emit(True)
                        print('Faces detected')
                    else:
                        self.frontalFaceDetection.emit(False)
                        print('No faces detected')

                # convert the image to QImage format
                # 将图像转换为QImage格式
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(400, 300, Qt.KeepAspectRatio)

                if p is not None and self.imageQ.qsize() < 2:
                    # put the image into image queue buffer
                    # 将图像放入图像队列缓冲区
                    self.imageQ.put(p)
                else:
                    # sleep if there is no valid image or the buffer is full
                    # 在没有有效的图像或缓冲区已满时休眠
                    time.sleep(DISP_MSEC / 100.0)
        
class CameraWindow(QMainWindow):
    def __init__(self, faceDetectionThresholdInSec):
        # create camera data window
        # 创建相机数据窗口
        super(CameraWindow, self).__init__()
        self.setGeometry(0,0,400,300)
        self.setWindowTitle("Video Streaming")

        # create the QLabel for the video frame to be drawn on
        # 为要绘制的视频帧创建QLabel
        self.label = QLabel()
        self.label.move(0, 0)
        self.label.resize(400, 300)
        self.label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.label)

        # create a timer to draw camera captured frame from memory queue
        # 创建一个计时器以从内存队列中绘制相机捕获的帧
        self.timer = QTimer(self)
        self.timer.timeout.connect(lambda: self.show_image(image_queue))
        self.timer.start(DISP_MSEC)
        
        # create the thread to capture image with camera and detecting human face
        # 创建线程以使用相机捕获图像并检测人脸
        self.th = CameraCaptureThread(faceDetectionThresholdInSec, image_queue)
        # connect the signal from thread to its recipient
        # 将信号从线程连接到其接收者
        self.th.frontalFaceDetection.connect(videoWindow.playSignalFromFrontalFaceDetection)
        self.th.start()
        self.show()

    def closeEvent(self, event):
        # safely destruct the QThread
        # 安全地销毁QThread
        self.th.quit()
        # stop the QTimer
        # 停止QTimer
        self.timer.stop()
        app.closeAllWindows()

    # fetch camera image from memory queue, and display it on QLabel
    # 从内存队列中获取相机图像，并将其显示在QLabel上
    def show_image(self, imageq):
        if not imageq.empty():
            image = imageq.get()
            if image is not None:
                self.label.setPixmap(QPixmap.fromImage(image))

class VideoWindow(QMainWindow):
    def __init__(self):
        # create video window
        # 视频窗口创建
        super(VideoWindow, self).__init__()
        self.setWindowTitle("Genius' creation")
        self.move(500, 0)
        self.resize(640, 480)

        # create a VLC media player software object
        # 创建一个VLC媒体播放器
        self.instance = vlc.Instance()
        self.mediaPlayer = self.instance.media_player_new()
        # create a QFrame for the video frame to be drawn on
        # 创建一个QFrame以绘制视频的帧
        self.videoframe = QFrame()
        self.palette = self.videoframe.palette()
        self.palette.setColor (QtGui.QPalette.Window,
                               QtGui.QColor(0,0,0))
        self.videoframe.setPalette(self.palette)
        self.videoframe.setAutoFillBackground(True)
        
        # create play and pause button using QPushButton
        # 使用QPushButton创建播放和暂停按钮
        self.playButton = QPushButton()
        self.playButton.setEnabled(True)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)

        # create video playing progress slider using QSlider
        # 使用QSlider创建视频播放进度滑块
        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setToolTip("Position")
        self.positionSlider.setMaximum(1000)
        self.positionSlider.sliderMoved.connect(self.setPosition)

        # create a QWidget for all the above user interface contents
        # 为上述所有用户界面内容创建一个QWidget
        wid = QWidget(self)
        self.setCentralWidget(wid)

        # create layouts to place inside QWidget
        # 在QWidget中创建用户界面布局
        controlLayout = QHBoxLayout()
        controlLayout.setContentsMargins(0, 0, 0, 0)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.positionSlider)

        layout = QVBoxLayout()
        layout.addWidget(self.videoframe)
        layout.addLayout(controlLayout)

        # set QWidget to contain the created layouts
        # 设置QWidget以包含创建的用户界面布局
        wid.setLayout(layout)
        
        # start a timer to update the user interface in interval of 200ms
        # 启动计时器以200毫秒的间隔更新用户界面
        self.timer = QTimer(self)
        self.timer.timeout.connect(lambda: self.updateUI())
        self.timer.start(200)
        
        # load the video file from the running script path
        # 从正在运行的脚本路径加载视频文件
        self.media = self.instance.media_new(os.getcwd() + '/APAR_clip.mp4')
        self.mediaPlayer.set_media(self.media)
        # parse the metadata of the file
        # 解析文件的元数据
        self.media.parse()
        # set the title of the video as video window title
        # 将视频的标题设置为视频窗口标题
        self.setWindowTitle(self.media.get_meta(0))

        # set the VLC media player frame redrawn handler according to operating system
        # 根据操作系统设置VLC媒体播放器框架重绘处理程序
        if sys.platform.startswith('linux'):  # for Linux using the X Server
            self.mediaPlayer.set_xwindow(self.videoframe.winId())
        elif sys.platform == "win32":  # for Windows
            self.mediaPlayer.set_hwnd(self.videoframe.winId())
            pass
        elif sys.platform == "darwin":  # for MacOS
            self.mediaPlayer.set_nsobject(self.videoframe.winId())
        
        self.show()

    # software routine to update user interface
    # 用于更新用户界面的软件例程
    def updateUI (self):
        # update the position slider according to the video playing progress
        # 根据视频播放进度更新播放位置滑块
        self.positionSlider.setValue(self.mediaPlayer.get_position() * 1000)

    # software action to play button being click
    # 点击播放按钮的软件动作
    def play(self):
        if self.mediaPlayer.is_playing():
            # when media player is playing, pause the video
            # 媒体播放器正在播放中，暂停视频
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            self.mediaPlayer.pause()
        else:
            # when media player is pausing, play the video
            # 媒体播放器在暂停中，播放视频
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.mediaPlayer.play()

    # software routine to handle signal from camera data window
    # 处理来自摄像机数据窗口的信号的软件例程
    def playSignalFromFrontalFaceDetection(self, detected):
        # when no human face is detected
        # 没有检测到人脸时
        if self.mediaPlayer.is_playing() and detected == False:
            # media player pause
            # 媒体播放器暂停
            self.mediaPlayer.pause()
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        # when human face is detected
        # 没有检测到人脸时
        elif self.mediaPlayer.is_playing() == False and detected == True:
            # media player play
            # 媒体播放器播放
            self.mediaPlayer.play()
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    # set the video clip playing position
    # 设置视频片段的播放位置
    def setPosition(self, position):
        self.mediaPlayer.set_position(position / 1000.0)

    def closeEvent(self, event):
        app.closeAllWindows()


if __name__ == '__main__':
    # starting the program
    # 启动程序
    app = QApplication(sys.argv)

    # create video window
    # 视频窗口创建
    videoWindow = VideoWindow()
    # create camera data window
    # 相机数据窗口创建
    cameraWindow = CameraWindow(1)

    sys.exit(app.exec_())
