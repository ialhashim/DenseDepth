import os
import sys
import glob
import time
import math
import argparse
import numpy as np

# Computer Vision
import cv2
from scipy import ndimage
from skimage.transform import resize

# Visualization
import matplotlib.pyplot as plt
plasma = plt.get_cmap('plasma')

# UI and OpenGL
from PySide2 import QtCore, QtGui, QtWidgets, QtOpenGL
from OpenGL import GL, GLU
from OpenGL.arrays import vbo
from OpenGL.GL import shaders
import glm

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
args = parser.parse_args()

# Image shapes
height_rgb, width_rgb = 480, 640
height_depth, width_depth = height_rgb // 2, width_rgb // 2
rgb_width = width_rgb
rgb_height = height_rgb

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
global graph,model
graph = tf.compat.v1.get_default_graph()

def load_model():
    # Kerasa / TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
    from keras.models import load_model
    from layers import BilinearUpSampling2D

    # Custom object needed for inference and training
    custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

    # Load model into GPU / CPU
    return load_model(args.model, custom_objects=custom_objects, compile=False)

# Function timing
ticTime = time.time()
def tic(): global ticTime; ticTime = time.time()
def toc(): print('{0} seconds.'.format(time.time() - ticTime))

# Conversion from Numpy to QImage and back
def np_to_qimage(a):
    im = a.copy()
    return QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888).copy()
    
def qimage_to_np(img):
    img = img.convertToFormat(QtGui.QImage.Format.Format_ARGB32)
    return np.array(img.constBits()).reshape(img.height(), img.width(), 4)

# Compute edge magnitudes
def edges(d):
    dx = ndimage.sobel(d, 0)  # horizontal derivative
    dy = ndimage.sobel(d, 1)  # vertical derivative
    return np.abs(dx) + np.abs(dy)

# Main window
class Window(QtWidgets.QWidget):
    updateInput = QtCore.Signal()

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.model = None
        self.capture = None
        self.glWidget = GLWidget()

        mainLayout = QtWidgets.QVBoxLayout()

        # Input / output views
        viewsLayout = QtWidgets.QGridLayout()        
        self.inputViewer = QtWidgets.QLabel("[Click to start]")
        self.inputViewer.setPixmap(QtGui.QPixmap(rgb_width,rgb_height))
        self.outputViewer = QtWidgets.QLabel("[Click to start]")
        self.outputViewer.setPixmap(QtGui.QPixmap(rgb_width//2,rgb_height//2))

        imgsFrame = QtWidgets.QFrame()        
        inputsLayout = QtWidgets.QVBoxLayout()  
        imgsFrame.setLayout(inputsLayout)
        inputsLayout.addWidget(self.inputViewer)
        inputsLayout.addWidget(self.outputViewer)

        viewsLayout.addWidget(imgsFrame,0,0)
        viewsLayout.addWidget(self.glWidget,0,1)
        viewsLayout.setColumnStretch(1, 10)
        mainLayout.addLayout(viewsLayout)

        # Load depth estimation model      
        toolsLayout = QtWidgets.QHBoxLayout()  

        self.button = QtWidgets.QPushButton("Load model...")
        self.button.clicked.connect(self.loadModel)
        toolsLayout.addWidget(self.button)

        self.button5 = QtWidgets.QPushButton("Load image")
        self.button5.clicked.connect(self.loadImageFile)
        toolsLayout.addWidget(self.button5)

        self.button2 = QtWidgets.QPushButton("Webcam")
        self.button2.clicked.connect(self.loadCamera)
        toolsLayout.addWidget(self.button2)

        self.button3 = QtWidgets.QPushButton("Video")
        self.button3.clicked.connect(self.loadVideoFile)
        toolsLayout.addWidget(self.button3)

        self.button4 = QtWidgets.QPushButton("Pause")
        self.button4.clicked.connect(self.loadImage)
        toolsLayout.addWidget(self.button4)

        self.button6 = QtWidgets.QPushButton("Refresh")
        self.button6.clicked.connect(self.updateCloud)
        toolsLayout.addWidget(self.button6)

        mainLayout.addLayout(toolsLayout)        

        self.setLayout(mainLayout)
        self.setWindowTitle(self.tr("RGBD Viewer"))

        # Signals
        self.updateInput.connect(self.update_input)

        # Default example
        img = (self.glWidget.rgb * 255).astype('uint8')
        self.inputViewer.setPixmap(QtGui.QPixmap.fromImage(np_to_qimage(img)))
        coloredDepth = (plasma(self.glWidget.depth[:,:,0])[:,:,:3] * 255).astype('uint8')
        self.outputViewer.setPixmap(QtGui.QPixmap.fromImage(np_to_qimage(coloredDepth)))
        
    def loadModel(self):        
        QtGui.QGuiApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        tic()  
        self.model = load_model()
        print('Model loaded.')
        toc()
        self.updateCloud()
        QtGui.QGuiApplication.restoreOverrideCursor()

    def loadCamera(self):        
        self.capture = cv2.VideoCapture(0)
        self.updateInput.emit()

    def loadVideoFile(self):
        self.capture = cv2.VideoCapture('video.mp4')
        self.updateInput.emit()

    def loadImage(self):
        self.capture = None
        img = (self.glWidget.rgb * 255).astype('uint8')
        self.inputViewer.setPixmap(QtGui.QPixmap.fromImage(np_to_qimage(img)))
        self.updateCloud()

    def loadImageFile(self):
        self.capture = None
        filename = QtWidgets.QFileDialog.getOpenFileName(None, 'Select image', '', self.tr('Image files (*.jpg *.png)'))[0]
        img = QtGui.QImage(filename).scaledToHeight(rgb_height)
        xstart = 0
        if img.width() > rgb_width: xstart = (img.width() - rgb_width) // 2
        img = img.copy(xstart, 0, xstart+rgb_width, rgb_height)
        self.inputViewer.setPixmap(QtGui.QPixmap.fromImage(img))
        self.updateCloud()

    def update_input(self):
        # Don't update anymore if no capture device is set
        if self.capture == None: return

        # Capture a frame
        ret, frame = self.capture.read()

        # Loop video playback if current stream is video file
        if not ret:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.capture.read()

        # Prepare image and show in UI
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = np_to_qimage(frame)
        self.inputViewer.setPixmap(QtGui.QPixmap.fromImage(image))

        # Update the point cloud
        self.updateCloud()

    def updateCloud(self):
        rgb8 = qimage_to_np(self.inputViewer.pixmap().toImage())
        self.glWidget.rgb = resize((rgb8[:,:,:3]/255)[:,:,::-1], (rgb_height, rgb_width), order=1, anti_aliasing=True)

        if self.model: 
            with graph.as_default():
                depth = (1000 / self.model.predict( np.expand_dims(self.glWidget.rgb, axis=0)  )) / 1000
            coloredDepth = (plasma(depth[0,:,:,0])[:,:,:3] * 255).astype('uint8')
            self.outputViewer.setPixmap(QtGui.QPixmap.fromImage(np_to_qimage(coloredDepth)))
            self.glWidget.depth = depth[0,:,:,0]
        else:
            self.glWidget.depth = 0.5 + np.zeros((rgb_height//2, rgb_width//2, 1))
            
        self.glWidget.updateRGBD()
        self.glWidget.updateGL()

        # Update to next frame if we are live
        QtCore.QTimer.singleShot(10, self.updateInput)

class GLWidget(QtOpenGL.QGLWidget):
    def __init__(self, parent=None):
        QtOpenGL.QGLWidget.__init__(self, parent)

        self.object = 0
        self.xRot = 5040
        self.yRot = 40
        self.zRot = 0
        self.zoomLevel = 9

        self.lastPos = QtCore.QPoint()

        self.trolltechGreen = QtGui.QColor.fromCmykF(0.40, 0.0, 1.0, 0.0)
        self.trolltechPurple = QtGui.QColor.fromCmykF(0.39, 0.39, 0.0, 0.0)

        # Precompute for world coordinates 
        self.xx, self.yy = self.worldCoords(width=rgb_width//2, height=rgb_height//2)

        # Load test frame from disk
        self.rgb = np.load('demo_rgb.npy')
        self.depth = resize(np.load('demo_depth.npy'), (240,320))
        self.col_vbo = None
        self.pos_vbo = None
        self.updateRGBD()

    def xRotation(self):
        return self.xRot

    def yRotation(self):
        return self.yRot

    def zRotation(self):
        return self.zRot

    def minimumSizeHint(self):
        return QtCore.QSize(640, 480)

    def sizeHint(self):
        return QtCore.QSize(640, 480)

    def setXRotation(self, angle):
        if angle != self.xRot:
            self.xRot = angle
            self.emit(QtCore.SIGNAL("xRotationChanged(int)"), angle)
            self.updateGL()

    def setYRotation(self, angle):
        if angle != self.yRot:
            self.yRot = angle
            self.emit(QtCore.SIGNAL("yRotationChanged(int)"), angle)
            self.updateGL()

    def setZRotation(self, angle):
        if angle != self.zRot:
            self.zRot = angle
            self.emit(QtCore.SIGNAL("zRotationChanged(int)"), angle)
            self.updateGL()

    def resizeGL(self, width, height):
        GL.glViewport(0, 0, width, height)            

    def mousePressEvent(self, event):
        self.lastPos = QtCore.QPoint(event.pos())

    def mouseMoveEvent(self, event):
        dx = -(event.x() - self.lastPos.x())
        dy = (event.y() - self.lastPos.y())

        if event.buttons() & QtCore.Qt.LeftButton:
            self.setXRotation(self.xRot + dy)
            self.setYRotation(self.yRot + dx)
        elif event.buttons() & QtCore.Qt.RightButton:
            self.setXRotation(self.xRot + dy)
            self.setZRotation(self.zRot + dx)

        self.lastPos = QtCore.QPoint(event.pos())

    def wheelEvent(self, event):
        numDegrees = event.delta() / 8
        numSteps = numDegrees / 15
        self.zoomLevel = self.zoomLevel + numSteps
        event.accept()
        self.updateGL()

    def initializeGL(self):
        self.qglClearColor(self.trolltechPurple.darker())
        GL.glShadeModel(GL.GL_FLAT)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_CULL_FACE)

        VERTEX_SHADER = shaders.compileShader("""#version 330
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 color;
        uniform mat4 mvp; out vec4 frag_color;
        void main() {gl_Position = mvp * vec4(position, 1.0);frag_color = vec4(color, 1.0);}""", GL.GL_VERTEX_SHADER)

        FRAGMENT_SHADER = shaders.compileShader("""#version 330
        in vec4 frag_color; out vec4 out_color; 
        void main() {out_color = frag_color;}""", GL.GL_FRAGMENT_SHADER)

        self.shaderProgram = shaders.compileProgram(VERTEX_SHADER, FRAGMENT_SHADER)
        
        self.UNIFORM_LOCATIONS = {
            'position': GL.glGetAttribLocation( self.shaderProgram, 'position' ),
            'color': GL.glGetAttribLocation( self.shaderProgram, 'color' ),
            'mvp': GL.glGetUniformLocation( self.shaderProgram, 'mvp' ),
        }

        shaders.glUseProgram(self.shaderProgram)

    def paintGL(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        self.drawObject()

    def worldCoords(self, width, height):
        hfov_degrees, vfov_degrees = 57, 43
        hFov = math.radians(hfov_degrees)
        vFov = math.radians(vfov_degrees)
        cx, cy = width/2, height/2
        fx = width/(2*math.tan(hFov/2))
        fy = height/(2*math.tan(vFov/2))
        xx, yy = np.tile(range(width), height), np.repeat(range(height), width)
        xx = (xx-cx)/fx
        yy = (yy-cy)/fy
        return xx, yy

    def posFromDepth(self, depth):
        length = depth.shape[0] * depth.shape[1]

        depth[edges(depth) > 0.3] = 1e6  # Hide depth edges       
        z = depth.reshape(length)

        return np.dstack((self.xx*z, self.yy*z, z)).reshape((length, 3))

    def createPointCloudVBOfromRGBD(self):
        # Create position and color VBOs
        self.pos_vbo = vbo.VBO(data=self.pos, usage=GL.GL_DYNAMIC_DRAW, target=GL.GL_ARRAY_BUFFER)
        self.col_vbo = vbo.VBO(data=self.col, usage=GL.GL_DYNAMIC_DRAW, target=GL.GL_ARRAY_BUFFER)

    def updateRGBD(self):
        # RGBD dimensions
        width, height = self.depth.shape[1], self.depth.shape[0]

        # Reshape
        points = self.posFromDepth(self.depth.copy())
        colors = resize(self.rgb, (height, width)).reshape((height * width, 3))

        # Flatten and convert to float32
        self.pos = points.astype('float32')
        self.col = colors.reshape(height * width, 3).astype('float32')

        # Move center of scene
        self.pos = self.pos + glm.vec3(0, -0.06, -0.3)

        # Create VBOs
        if not self.col_vbo:
            self.createPointCloudVBOfromRGBD()

    def drawObject(self):
        # Update camera
        model, view, proj = glm.mat4(1), glm.mat4(1), glm.perspective(45, self.width() / self.height(), 0.01, 100)        
        center, up, eye = glm.vec3(0,-0.075,0), glm.vec3(0,-1,0), glm.vec3(0,0,-0.4 * (self.zoomLevel/10))
        view = glm.lookAt(eye, center, up)
        model = glm.rotate(model, self.xRot / 160.0, glm.vec3(1,0,0))
        model = glm.rotate(model, self.yRot / 160.0, glm.vec3(0,1,0))
        model = glm.rotate(model, self.zRot / 160.0, glm.vec3(0,0,1))
        mvp = proj * view * model
        GL.glUniformMatrix4fv(self.UNIFORM_LOCATIONS['mvp'], 1, False, glm.value_ptr(mvp))

        # Update data
        self.pos_vbo.set_array(self.pos)
        self.col_vbo.set_array(self.col)

        # Point size
        GL.glPointSize(4)

        # Position
        self.pos_vbo.bind()
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

        # Color
        self.col_vbo.bind()
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

        # Draw
        GL.glDrawArrays(GL.GL_POINTS, 0, self.pos.shape[0])

        # Center debug
        if False:
            self.qglColor(QtGui.QColor(255,0,0))
            GL.glPointSize(20)
            GL.glBegin(GL.GL_POINTS)
            GL.glVertex3d(0,0,0)
            GL.glEnd()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    res = app.exec_()