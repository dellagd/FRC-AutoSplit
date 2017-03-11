import cv2
import numpy as np
from functools import reduce

class webcastOCR:
    def __init__(self):
        samples = np.loadtxt('generalsamples.data',np.float32)
        responses = np.loadtxt('generalresponses.data',np.float32)
        responses = responses.reshape((responses.size,1))

        self.model = cv2.ml.KNearest_create()
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def getDigits(self, img, xS, yS, wS=32, hS=16):
        digitsList = []
        
        im = img[yS:yS+hS, xS:xS+wS]
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        foo, thresh = cv2.threshold(gray,210,255,cv2.THRESH_BINARY_INV)
        kernel = np.ones((2,2), np.uint8)
        thresh = cv2.erode(thresh, kernel, iterations=1)

        foo, contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt)>10:
                [x,y,w,h] = cv2.boundingRect(cnt)
                if  h>8:
                    cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
                    roi = thresh[y:y+h,x:x+w]
                    roismall = cv2.resize(roi,(10,10))
                    roismall = roismall.reshape((1,100))
                    roismall = np.float32(roismall)
                    retval, results, neigh_resp, dists = self.model.findNearest(roismall, k = 1)
                    string = str(int((results[0][0])))
                    #cv2.putText(out,string,(x,y+h),0,.5,(0,255,0))
                    digitsList.append((x, int(results[0][0])))

        
        digitsList.sort(key=lambda x: x[0])

        if (len(digitsList) >= 2):
            i = 1;
            lim = len(digitsList)
            while i < lim:
                if abs(digitsList[i-1][0] - digitsList[i][0]) < 3:
                    del digitsList[i-1]
                    lim -= 1
                else:
                    i += 1
        
        p = reduce(lambda x,y: x+str(y[1]), digitsList, "")
        #print(p)
        return p



