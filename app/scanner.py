import cv
import time
import threading
import subprocess
import os
import os.path

class Scanner(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.event = threading.Event()
        self.running = True
        self.capturing = False
        self.notify = None
        self.scanner = None
        self.captured_frames = 0

    def image(self):
        if self.last_image != None:
            image = self.last_image
            self.last_image = None # make sure we don't return duplicate images
            return image
        return None

    def scan(self):
        filename = "/tmp/scan.tif"
        if os.path.exists(filename):
            os.remove(filename)
        proc = subprocess.Popen(("scanimage --batch=%s --batch-count=1 --resolution=600 --mode=color --format=tiff" % filename).split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out,err = proc.communicate()
        if os.path.exists(filename):
            return cv.LoadImage(filename)
        else:
            print "error", err
            return None

    def run(self):
        while self.running:
            try:
                if self.capturing:
                    image = self.scan()
                    if not image:
                        time.sleep(10)
                    else:
                        self.last_image = image
                        if self.notify:
                            self.notify.set()
                        time.sleep(0.2) # yield to other threads

                    # automatically stop capturing after some number of scans
                    if self.captured_frames > 10:
                        self.capturing = False
                    else:
                        self.captured_frames += 1
                else:
                    self.event.wait()
                    self.event.clear()
            except Exception, e:
                print e
                time.sleep(10)

    def exit_thread(self):
        self.running = False
        self.event.set()

    def start_scan(self, notify):
        self.notify = notify
        self.capturing = True
        self.captured_frames = 0
        self.event.set()

    def stop_scan(self):
        self.notify = None
        self.capturing = False
        self.last_image = None
        self.event.set()
