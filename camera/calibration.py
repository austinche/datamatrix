import sys
import cv
from params import Params

class Calibration:
    MAX_IMAGE_HEIGHT = 300
    def __init__(self, img):
        scale = 1
        if img.height > self.MAX_IMAGE_HEIGHT:
            scale = img.height / self.MAX_IMAGE_HEIGHT

        resized = cv.CreateImage((int(img.width / scale), int(img.height / scale)), img.depth, img.nChannels)
        cv.Resize(img, resized)
        self.click = None
        self.threshold_hue = Params.flood_fill_hue
        self.threshold_sat = Params.flood_fill_sat
        self.image = resized
        
        (self.hsv, self.hue, self.sat) = self.image2hsv(self.image)
        self.selected = cv.CreateImage(cv.GetSize(self.image), self.image.depth, self.image.nChannels)

    def calibrate(self):
        cv.NamedWindow("source")
        cv.CreateTrackbar("Hue", "source", self.threshold_hue, 255, self.update_hue)
        cv.CreateTrackbar("Sat", "source", self.threshold_sat, 255, self.update_sat)
        cv.ShowImage("source", self.image)
        cv.SetMouseCallback("source", self.mouse, 0)

        self.update_image()
        while True:
            k = cv.WaitKey(0)
            if k == ord('q'):
                break
            elif k == ord('s'):
                Params.save()
                print "Histograms saved to file"
            elif k == ord('b'):
                Params.box_histogram = self.histogram()
                print "Box histogram set"
            elif k == ord('t'):
                Params.tab_histogram = self.histogram()
                print "Tab histogram set"                

    def image2hsv(self, img):
        # convert standard image to HSV color space and returns hue/saturation channels
        hsv = cv.CreateImage(cv.GetSize(img), 8, 3)
        cv.CvtColor(img, hsv, cv.CV_BGR2HSV)
        hue = cv.CreateImage(cv.GetSize(hsv), cv.IPL_DEPTH_8U, 1)
        sat = cv.CreateImage(cv.GetSize(hsv), cv.IPL_DEPTH_8U, 1)
        cv.Split(hsv, hue, sat, None, None)
        return (hsv, hue, sat)

    def mouse(self, event, x, y, flags, param):
        if event != cv.CV_EVENT_LBUTTONDOWN:
            return

        self.click = (x, y)
        self.update_image()

    def update_image(self):
        if self.click == None:
            return

        self.mask = cv.CreateMat(self.hsv.height + 2, self.hsv.width + 2, cv.CV_8UC1)
        cv.SetZero(self.mask)
        cv.SetZero(self.selected)

        cv.FloodFill(self.hsv, self.click, (255, 255, 255), (self.threshold_hue, self.threshold_sat, 255), (self.threshold_hue, self.threshold_sat, 255), cv.CV_FLOODFILL_FIXED_RANGE + cv.CV_FLOODFILL_MASK_ONLY, self.mask)

        self.mask = cv.GetSubRect(self.mask, (1, 1, self.hsv.width, self.hsv.height))
        cv.Copy(self.image, self.selected, self.mask)

        cv.ShowImage("selected", self.selected)
        
    def update_hue(self, val):
        self.threshold_hue = val
        self.update_image()
    
    def update_sat(self, val):
        self.threshold_sat = val
        self.update_image()
    
    def histogram(self):
        # hue varies from 0 to 179
        # saturation varies from 0 (black-gray-white) to 255 (pure spectrum color)
        hist = cv.CreateHist([Params.hue_bins, Params.sat_bins], cv.CV_HIST_ARRAY, [[0, 179], [0, 255]])
        cv.CalcHist([self.hue, self.sat], hist, mask=self.mask)
        cv.NormalizeHist(hist, 255)
        return hist


if __name__ == "__main__":
    if len(sys.argv) > 1:
        img = cv.LoadImage(sys.argv[1], cv.CV_LOAD_IMAGE_COLOR)
        Calibration(img).calibrate()
    else:
        print "Give image to use for calibration"
