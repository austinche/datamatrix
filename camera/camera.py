import cv
import time
import threading

from decoder import BoxScanner

class Camera:
    def __init__(self):
        self.find_camera()

    def __del__(self):
        del(self.capture)

    def find_camera(self):
        self.capture = cv.CaptureFromCAM(0)
        frame = cv.QueryFrame(self.capture)
        if not frame:
            self.camera_present = False
            return
        
        self.camera_present = True
        # first frames are usually bad so we skip a couple
        for i in range(3):
            frame = cv.QueryFrame(self.capture)
        
    def frame(self):
        return cv.QueryFrame(self.capture)

    def present(self):
        return self.camera_present
    
class CameraThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.event = threading.Event()
        self.camera = Camera()
        self.box = None
        self.attempt_count = 0
        self.running = True

    def run(self):
        while self.running:
            try:
                if not self.camera.present():
                    self.camera.find_camera()
                if self.camera.present() and self.box != None and self.attempt_count < Params.max_box_scan_attempts:
                    self.attempt_count += 1
                    frame = self.camera.frame()
                    if not frame:
                        print "bad frame from camera"
                        continue
                    done = self.box.scan(frame)
                    if done:
                        self.box = None

                    time.sleep(Params.camera_sleep_between_pictures)
                else:
                    self.event.wait()
                    self.event.clear()
            except Exception, e:
                print e

    def exit(self):
        self.running = False
        self.event.set()
        
    def stop_box(self):
        self.box = None
        
    def start_box(self, box):
        self.box = box
        self.attempt_count = 0
        self.event.set()
