import cv
import time
import threading

from decoder import BoxScanner
from params import Params

class Camera(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.event = threading.Event()
        self.running = True
        self.capturing = False
        self.last_frame = None
        self.captured_frames = 0

    def __del__(self):
        del(self.capture)

    def find_camera(self):
        self.capture = cv.CaptureFromCAM(0)
        self.last_frame = cv.QueryFrame(self.capture)
        if not self.last_frame:
            self.camera_present = False
        else:
            self.camera_present = True

    def frame(self):
        if self.last_frame != None:
            frame = self.last_frame
            self.last_frame = None # make sure we don't return duplicate frames
            return cv.CloneImage(frame) # should not modify the original according to the docs
        return None

    def run(self):
        # queryframe seems to buffer frame and doesn't always return the latest when asked
        # the purpose of this thread is to continue grabbing frames to empty the buffer
        # even if we aren't going to use the frames
        self.find_camera()
        while self.running:
            try:
                if not self.camera_present:
                    self.find_camera()
                    time.sleep(10)
                elif self.capturing:
                    if self.captured_frames == 0:
                        # throw away some frames when starting a capture
                        # to avoid getting old frames
                        for i in range(5):
                            cv.QueryFrame(self.capture)

                    self.last_frame = cv.QueryFrame(self.capture)
                    # automatically stop capturing after some number of frames
                    if self.captured_frames > 1000:
                        self.capturing = False
                    else:
                        self.captured_frames += 1
                    time.sleep(0.2) # yield to other threads
                else:
                    self.event.wait()
                    self.event.clear()
            except Exception, e:
                print e

    def exit_thread(self):
        self.running = False
        self.event.set()

    def start_capture(self):
        self.capturing = True
        self.captured_frames = 0
        self.event.set()

    def stop_capture(self):
        self.capturing = False
        self.last_frame = None # throw away any existing frame
        self.event.set()

