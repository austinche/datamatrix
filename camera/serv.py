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
                        time.sleep(0.5)
                        continue

                    # shrink image
                    cv.ResetImageROI(image)

                    # image can be different sizes depending on if it's been cropped yet
                    # we shrink to fixed size and add a caption
                    shrink = cv.CreateImage((320, 280), image.depth, image.nChannels)
                    cv.SetImageROI(shrink, (0, 0, 320, 240))
                    cv.Resize(image, shrink)

                    # add caption text
                    cv.SetImageROI(shrink, (0, 240, 320, 40))
                    cv.Set(shrink, (255, 255, 255))
                    (count, empty) = MyHandler.box.decode_info
                    cv.PutText(shrink, "Empty: %d " % empty, (0, 20), MyHandler.font, (255, 0, 0))
                    cv.PutText(shrink, "Codes: %d " % (count - empty), (100, 20), MyHandler.font, (0, 255, 0))
                    cv.PutText(shrink, "Unknown: %d " % (96 - count), (200, 20), MyHandler.font, (0, 0, 255))

                    # switch from BGR to RGB
                    cv.ResetImageROI(shrink)
                    cv.CvtColor(shrink, shrink, cv.CV_BGR2RGB)

                    # use PIL to make jpeg
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

        except Exception, e:
            print "Exception in server thread"
            print e
            MyHandler.camera.stop_box()

        finally:
            try:
                self.wfile.close()
            except:
                pass

def main():
    MyHandler.box = BoxScanner()
    MyHandler.camera = camera.CameraThread()
    MyHandler.camera.start()
    MyHandler.font = cv.InitFont(cv.CV_FONT_HERSHEY_PLAIN, 1.0, 1.0)

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
