import sys
import numpy as np
import math
import cv2
import subprocess
import time
import webcastUtils
import csv

ocr = webcastUtils.webcastOCR()

def almostEqual(d1, d2):
    epsilon = 0.001
    return (abs(d2 - d1) < epsilon)

# Returns length of video in seconds
def getVideoLength(path):
    output = subprocess.check_output(
            ['ffprobe', path],
            stderr=subprocess.STDOUT)
    output = output.decode().splitlines()
    output = list(map(lambda x: x.strip(), output))
    output = list(filter(lambda x: "Duration" in x, output))
    res = output[0].split(',')[0].split(" ")[1] # Length timestamp
    timestamp = res.split(":")
    hrs = timestamp[0]
    mins = timestamp[1]
    secs = timestamp[2]
    totalsecs = int(hrs)*3600 + int(mins)*60 + math.floor(float(secs))
    print("Timestamp: %s, Total seconds: %d" % (res, totalsecs))
    return totalsecs    

def genFrame(path, timeSec):
    #st = time.time()
    output = subprocess.check_output(
            ['ffmpeg', '-ss', makeTimestamp(timeSec), '-i', path, '-vframes',
                '1',  '/home/griffin/Downloads/out.png', '-y'],
            stderr=subprocess.STDOUT)
    #print("Process time ffmpeg: %.3f uS" % ((time.time() - st)*1e6))
    #print(output.decode())

def makeTimestamp(secs):
    hrs = secs//3600
    mins = secs//60 - (hrs*60)
    secs = secs - (mins*60) - (hrs*3600)
    if almostEqual(secs, 60):
        mins += 1
        secs = 0
    if almostEqual(mins, 60):
        hrs += 1
        mins = 0

    timestamp = "%02d:%02d:%02.3f" % (hrs, mins, secs)
    #print ("Made timestamp: %s" % timestamp)
    
    return timestamp

def processImage():
    #st = time.time()
    img = cv2.imread('/home/griffin/Downloads/out.png')
    img2 = img.copy()
    templ = cv2.imread('templ.png')
    #print(len(templ.shape[::-1]))
    w, h, foo = templ.shape[::-1]

    res = cv2.matchTemplate(img2, templ, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1]+h)

    #print("Process time: %.3f uS" % ((time.time() - st)*1e6))

    #print("Max Val: %.2f" % max_val)
    if (max_val > 9e6):
        num = ocr.getDigits(img, 281, 373) 
        #print("Match %s" % num)
        return True
    #cv2.rectangle(img, top_left, bottom_right, 255, 2)
    #cv2.imwrite('out2.png', img)
    return False


def lookForMatchInProgress():
    img = cv2.imread('/home/griffin/Downloads/out.png')
    imgT = img[400:400+20, 372:372+5]
    img2 = imgT.copy()
    templ = cv2.imread('templMatchGoing.png')
    w, h, foo = templ.shape[::-1]

    res = cv2.matchTemplate(img2, templ, cv2.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1]+h)

    #print(min_val)
    if (min_val < 1e4):
        num = ocr.getDigits(img, 281, 373) 
        return num
    return None

# Arguments
if (len(sys.argv) < 3): quit()
youtubeLink = sys.argv[1]
vidPath = sys.argv[2]

vidLengthSeconds = getVideoLength(vidPath)

print()
startTime = input("Start analysis at time: ")
startTime = 0 if startTime=="" else eval(startTime)
endTime = input("End analysis at time: ")
endTime = vidLengthSeconds if endTime=="" else eval(endTime)

print("\n<-----Begin rough time detectioni----->\n")


i = startTime#5*3600
matchesRough = []
while i < endTime:
    #cv2.destroyAllWindows()
    genFrame(vidPath, i)
    i += 30
    res = lookForMatchInProgress()
    if res != None:
        print("Found match %s around time %s" % (res, makeTimestamp(i)))
        print("Youtube: %s?t=%d"%(youtubeLink, i))
        matchesRough.append((res, i))
        i += 5*60

print("\n<-----Begin fine time detection----->\n")
matchesFine = []
for match in matchesRough:
    i = 45
    found = False
    while i < 120:
        genFrame(vidPath, match[1]-i)
        if processImage():
            print("Match %s starts at exactly %d seconds"%(match[0], match[1]-i))
            fulllink = "%s?t=%d"%(youtubeLink, match[1]-i-4) 
            print("Youtube: %s"%fulllink)
            matchesFine.append((match[0], match[1]-i-4, fulllink))
            found = True
            break
        i += 0.85
    if not found:
        print("Error! No fine time found for match %s"%match[0])

mf = open("matches.csv", 'wt')
wr = csv.writer(mf)
wr.writerow(('Match', 'Time', 'Link'))
for line in matchesFine:
    wr.writerow(line)

cv2.destroyAllWindows()
