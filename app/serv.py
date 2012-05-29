"""
Server module for Scantelope.
This is the main application module.

INTERFACE:
    http://localhost:3333{command}

replace {command} with:       in order to:

      /cam                     start a scan and show image of current camera
      /                        view CSV of most-recently decoded
"""

from SocketServer import ThreadingMixIn
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer

PORT=3333

import cv
import time
import threading
import Image
import StringIO

import camera
import scanner
from decoder import BoxScanner

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    pass

class ScanThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.event = threading.Event()
        self.camera = camera.Camera()
        self.camera.start()
        self.scanner = scanner.Scanner()
        self.scanner.start()
        self.box = BoxScanner()
        self.attempt_count = 0
        self.running = True
        self.scanning = False
        self.notify = None

    def run(self):
        while self.running:
            try:
                if self.scanning and self.attempt_count < 100:
                    sleep_time = 10
                    image = self.camera.frame()
                    if image:
                        sleep_time = 1
                        self.attempt_count += 1
                    else:
                        image = self.scanner.image()
                        self.attempt_count += 10
                        if image:
                            sleep_time = 5
                    if image:
                        count = self.box.scan(image)
                        if count == 96:
                            self.stop_scan()
                        if self.notify:
                            self.notify.set()
                    self.event.wait(sleep_time)
                else:
                    self.stop_scan()
                    self.event.wait()
                self.event.clear()
            except Exception, e:
                print e

    def exit_thread(self):
        self.running = False
        self.camera.exit_thread()
        self.scanner.exit_thread()
        self.event.set()

    def stop_scan(self):
        self.notify = None
        self.scanning = False
        self.camera.stop_capture()
        self.scanner.stop_scan()

    def ensure_running(self):
        self.scanning = True
        self.camera.start_capture(self.event)
        self.scanner.start_scan(self.event)
        self.event.set()

    def reset_box(self, notify):
        self.box = BoxScanner()
        self.attempt_count = 0
        self.notify = notify
        self.event.set()

    def image(self, width, height):
        image = self.box.annotated_image(width, height)
        if image == None:
            return (0, None)

        # switch from BGR to RGB
        cv.CvtColor(image, image, cv.CV_BGR2RGB)

        # use PIL to make jpeg
        im = Image.fromstring("RGB", cv.GetSize(image), image.tostring())
        f = StringIO.StringIO()
        im.save(f, "JPEG")
        data = f.getvalue()
        return (f.len, data)

    def write_box_info(self, wfile):
        self.box.write_box_info(wfile)

    def write_code_csv(self, wfile):
        self.box.write_code_csv(wfile)

class MyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        MJPEG_BOUNDARY="a34e78adc034c428d4a35c1831db7bf8" # random string
        try:
            if self.path.startswith("/cam"):
                # stream via motion JPEG
                # http://en.wikipedia.org/wiki/Motion_JPEG
                notify = threading.Event()
                MyHandler.scanner.reset_box(notify)
                MyHandler.scanner.ensure_running()
                self.send_response(200)
                self.send_header('Content-type', "multipart/x-mixed-replace;boundary=%s" % MJPEG_BOUNDARY)
                self.end_headers()
                for count in range(100): # limit number of images to show before stopping
                    (length, data) = MyHandler.scanner.image(320, 280)
                    # image may be unavailable
                    if length > 0:
                        self.wfile.write("--%s\r\n" % MJPEG_BOUNDARY)
                        self.send_header('Content-type', 'image/jpeg')
                        self.send_header('Content-length', length)
                        self.end_headers()
                        self.wfile.write(data)
                    notify.wait(1)
                    notify.clear()
                MyHandler.scanner.stop_scan()
            elif self.path.startswith("/text"):
                # stream text info
                notify = threading.Event()
                MyHandler.scanner.reset_box(notify)
                MyHandler.scanner.ensure_running()
                self.send_response(200)
                self.send_header('Content-type', "text/plain")
                self.end_headers()
                for count in range(100): # limit number of images to show before stopping
                    MyHandler.scanner.write_box_info(self.wfile)
                    notify.wait(1)
                    notify.clear()
                MyHandler.scanner.stop_scan()
            elif self.path.strip('/') == '':
                MyHandler.scanner.ensure_running()
                self.send_response(200)
                self.send_header('Content-type','text/plain')
                self.end_headers()
                codes = MyHandler.scanner.write_code_csv(self.wfile)

            else:
                self.send_error(404, 'File not found')

        except Exception, e:
            print "Exception in server thread"
            print e
            MyHandler.scanner.stop_scan()

def main():
    MyHandler.scanner = ScanThread()
    MyHandler.scanner.start()

    server = ThreadingHTTPServer(('', PORT), MyHandler)
    print 'started httpserver...'

    running = True
    try:
        while running:
            server.handle_request() # blocks until request
    except KeyboardInterrupt:
        print '^C received'
    finally:
        print 'Shutting down server'
        MyHandler.scanner.exit_thread()
        server.socket.close()

if __name__ == '__main__':
    main()
