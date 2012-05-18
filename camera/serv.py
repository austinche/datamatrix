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
import Image
import StringIO

import camera
from decoder import BoxScanner

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    pass

class MyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        MJPEG_BOUNDARY="a34e78adc034c428d4a35c1831db7bf8" # random string
        try:
            if self.path.startswith("/cam"):
                # stream via motion JPEG
                # http://en.wikipedia.org/wiki/Motion_JPEG
                MyHandler.box = BoxScanner()
                MyHandler.camera.start_box(MyHandler.box)
                self.send_response(200)
                self.send_header('Content-type', "multipart/x-mixed-replace;boundary=%s" % MJPEG_BOUNDARY)
                self.end_headers()
                for count in range(100): # limit number of images to show before stopping
                    image = MyHandler.box.last_image()
                    if image == None:
                        print "no image?"
                        continue
                    
                    # shrink image
                    cv.ResetImageROI(image)
                    shrink = cv.CreateImage((image.width / 8, image.height / 8), image.depth, image.nChannels)
                    cv.Resize(image, shrink)
                    
                    # switch from BGR to RGB
                    cv.CvtColor(shrink, shrink, cv.CV_BGR2RGB)
                    
                    # use PIL to resize/make jpeg
                    im = Image.fromstring("RGB", cv.GetSize(shrink), shrink.tostring())
                    f = StringIO.StringIO()
                    im.save(f, "JPEG")
                    data = f.getvalue()

                    self.wfile.write("--%s\r\n" % MJPEG_BOUNDARY)
                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Content-length', f.len)
                    self.end_headers()
                    self.wfile.write(data)
                    time.sleep(0.5)
                MyHandler.camera.stop_box()
                
            elif self.path.strip('/') == '':
                self.send_response(200)
                self.send_header('Content-type','text/plain')
                self.end_headers()
                codes = MyHandler.box.write_code_csv(self.wfile)
 
            else:
                self.send_error(404, 'File not found')
 
        except:
            MyHandler.camera.stop_box()
            
def main():
    MyHandler.box = BoxScanner()
    MyHandler.camera = camera.CameraThread()
    MyHandler.camera.start()
    
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
        MyHandler.camera.exit()
        server.socket.close()

if __name__ == '__main__':
    main()
