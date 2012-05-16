class Camera:
    def __init__(self):
        self.capture = cv.CaptureFromCAM(0)
        frame = cv.QueryFrame(self.capture)
        if not frame:
            raise "no camera found"

        # first frames are usually bad so we skip a couple
        for i in range(5):
            frame = cv.QueryFrame(self.capture)

    def __del__(self):
        del(self.capture)

    def frame(self):
        return cv.QueryFrame(self.capture)
