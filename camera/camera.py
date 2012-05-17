import cv
import threading

from params import Params
from decoder import BoxScanner

class Camera:
    def __init__(self):
        self.capture = cv.CaptureFromCAM(0)
        frame = cv.QueryFrame(self.capture)
        if not frame:
            raise "no camera found"

        # first frames are usually bad so we skip a couple
        for i in range(3):
            frame = cv.QueryFrame(self.capture)

    def __del__(self):
        del(self.capture)

    def frame(self):
        return cv.QueryFrame(self.capture)

class CameraThread(threading.Thread):
    def __init__(self):
        Params.load()
        threading.Thread.__init__(self)
        self.event = threading.Event()
        self.camera = Camera()
        self.box = None
        self.attempt_count = 0
        self.running = True
        
    def run(self):
        while self.running:
            if self.box != None and self.attempt_count < Params.max_box_scan_attempts:
                self.attempt_count += 1
                frame = self.camera.frame()
                done = self.box.scan(frame)
                if done:
                    self.box = None
            else:
                self.event.wait()
                self.event.clear()

    def stop(self):
        self.running = False
        self.event.set()
        
    def start_box(self, box):
        self.box = box
        self.attempt_count = 0
        self.event.set()
