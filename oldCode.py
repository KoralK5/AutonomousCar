import cv2
import numpy as np
import math

def makeCords(image, lineParameters):
    try:
        slope, intercept = lineParameters
        y1 = image.shape[0]
        y2 = int(y1*0.6)
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)
        if 0.2 > slope > -0.2:
            return np.array([shp.shape[1]/2, shp.shape[0]*3/5, shp.shape[1]/2, shp.shape[0]*3/5])
        else:
            return np.array([x1, y1, x2, y2])
    except:
        return np.array([shp.shape[1]/2, shp.shape[0]*3/5, shp.shape[1]/2, shp.shape[0]*3/5])

def avgSlopeInt(image, avgLines):
    leftFit = []
    rightFit = []

    for line in avgLines:
        x1, y1, x2, y2 = line.reshape(4)
        param = np.polyfit((x1, x2), (y1, y2), 1)
        slope = param[0]
        intercept = param[1]
        if slope < 0:
            leftFit.append((slope, intercept))
        else:
            rightFit.append((slope, intercept))
    
    leftLine = makeCords(image, np.average(leftFit, axis=0))
    rightLine = makeCords(image, np.average(rightFit, axis=0))
    return np.array([leftLine, rightLine])

def canny(image):
    grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurImg = cv2.GaussianBlur(grayImg, (5, 5), 0) #smoothens to 5x5 pixels
    cannyImg = cv2.Canny(blurImg, 50, 150) #increase 150 or decrease 50 to decrease line amount
    return cannyImg

def makePoly(image, lines):
    lineImg = np.zeros_like(image)
    if lines is not None:
        x1, y1, x2, y2 = lines[0]
        x3, y3, x4, y4 = lines[1]
        pts = np.array([[x2,y2],[x1,y1],[x4,y4],[x1,y1]], np.int32)
        pts = pts.reshape((-1,1,2))
        try:
            if x2 >= x4:
                cv2.fillPoly(lineImg,[pts[:-1]],(255,0,0))
            else:
                cv2.fillPoly(lineImg,[pts],(255,0,0))
        except:
            pass
    return lineImg

'''
def limit(image):
    height = image.shape[0]
    lenght = image.shape[1]
    polygons = np.array([[(0, height), (lenght, height), (int(lenght*3/5), int(height/2)), (int(lenght*2/5), int(height/2))]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    maskedImg = cv2.bitwise_and(image, mask)
    return maskedImg
'''

def limit(image):
    height = image.shape[0]
    width = image.shape[1]
    top_width = int(width/3)
    bottom_width = int(width*4/5)
    trapezoid_height = int(height/4)
    position = height - trapezoid_height

    # Create a black mask with the same dimensions as the original image
    mask = np.zeros((height, width), dtype=np.uint8)

    # Define the coordinates of the trapezoid
    points = np.array([
        [int((width - top_width) / 2), position],
        [int((width + top_width) / 2), position],
        [int((width + bottom_width) / 2), position + trapezoid_height],
        [int((width - bottom_width) / 2), position + trapezoid_height],
    ], dtype=np.int32)

    # Reshape the points array to the shape required by cv2.fillPoly
    points = points.reshape((-1, 1, 2))

    # Draw the trapezoid on the mask
    cv2.fillPoly(mask, [points], color=255)

    # Create the trapezoid cutout by applying the mask to the original image
    trapezoid_cutout = cv2.bitwise_and(image, image, mask=mask)

    return trapezoid_cutout



def steering(frame, lines, currAngle):
    height, width, _ = frame.shape
    try:
        _, _, leftX2, _ = lines[0]
    except:
        leftX2 = 1
    try:
        _, _, rightX2, _ = lines[1]
    except:
        rightX2 = shp.shape[0]

    mid = int(width / 2)
    xOffset = (leftX2 + rightX2) / 2 - mid
    yOffset = int(height / 2)
    angleToMidRadian = math.atan(xOffset / yOffset)
    angleToMidDeg = int(angleToMidRadian * 180.0 / math.pi)
    steeringAngle = angleToMidDeg + 90
    stabalizedAngle = stabilizeAngle(currAngle, steeringAngle, len(avgLines), 5, 1)
    currAngle = steeringAngle
    headingImage = directionLine(frame, stabalizedAngle, (0, 0, 255), 10)
    
    return headingImage, steeringAngle

def directionLine(frame, steeringAngle, lineColor, lineWidth):
    headingImg = np.zeros_like(frame)
    height, width, _ = frame.shape
    steeringAngleRadian = steeringAngle / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steeringAngleRadian))
    y2 = int(height*3/4)

    cv2.line(headingImg, (x1, y1), (x2, y2), lineColor, lineWidth)
    headingImg = cv2.addWeighted(frame, 0.8, headingImg, 1, 1)

    return headingImg

def stabilizeAngle(currSteeringAngle, newSteeringAngle, numLaneLines, maxAngleDeviation2Lines, maxAngleDeviation1Line):
    if numLaneLines == 2:
        maxAngleDeviation = maxAngleDeviation2Lines
    else :
        maxAngleDeviation = maxAngleDeviation1Line
    
    angleDeviation = newSteeringAngle - currSteeringAngle
    if abs(angleDeviation) > maxAngleDeviation:
        stabilizedSteeringAngle = int(currSteeringAngle + maxAngleDeviation * angleDeviation / abs(angleDeviation))
    else:
        stabilizedSteeringAngle = newSteeringAngle
    return stabilizedSteeringAngle

def colors(tripleImg, currAngle, textCenter, arrowCenter):
    tripleImg = cv2.rectangle(tripleImg, (shp.shape[1], 0), (shp.shape[1]-textCenter[0], textCenter[1]), (0, 0, 0), -1)
    if currAngle > 90:
        if currAngle <= 100:
            tripleImg = cv2.putText(tripleImg, str(currAngle-90), ((shp.shape[1]-textCenter[0]), textCenter[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            tripleImg = cv2.putText(tripleImg, '->', (int((shp.shape[1]-arrowCenter[0])*3/4), int(shp.shape[0]*2/3)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        elif 110 >= currAngle > 100:
            tripleImg = cv2.putText(tripleImg, str(currAngle-90), ((shp.shape[1]-textCenter[0]), textCenter[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            tripleImg = cv2.putText(tripleImg, '->', (int((shp.shape[1]-arrowCenter[0])*3/4), int(shp.shape[0]*2/3)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)
        elif currAngle > 110:
            tripleImg = cv2.putText(tripleImg, str(currAngle-90), ((shp.shape[1]-textCenter[0]), textCenter[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            tripleImg = cv2.putText(tripleImg, '->', (int((shp.shape[1]-arrowCenter[0])*3/4), int(shp.shape[0]*2/3)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    elif currAngle < 90:
        if currAngle >= 80:
            tripleImg = cv2.putText(tripleImg, str(currAngle-90), ((shp.shape[1]-textCenter[0]), textCenter[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            tripleImg = cv2.putText(tripleImg, '<-', (int((shp.shape[1]-arrowCenter[0])*1/4), int(shp.shape[0]*2/3)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        elif 70 <= currAngle < 80:
            tripleImg = cv2.putText(tripleImg, str(currAngle-90), ((shp.shape[1]-textCenter[0]), textCenter[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            tripleImg = cv2.putText(tripleImg, '<-', (int((shp.shape[1]-arrowCenter[0])*1/4), int(shp.shape[0]*2/3)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)
        elif currAngle < 70:
            tripleImg = cv2.putText(tripleImg, str(currAngle-90), ((shp.shape[1]-textCenter[0]), textCenter[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            tripleImg = cv2.putText(tripleImg, '<-', (int((shp.shape[1]-arrowCenter[0])*1/4), int(shp.shape[0]*2/3)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    else:
        tripleImg = cv2.putText(tripleImg, '0', ((shp.shape[1]-textCenter[0]), textCenter[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return tripleImg

while True:
    test = 'Videos/' + input('Test Video: ')
    cap = cv2.VideoCapture(test)
    _, shp = cap.read()
    currAngle = 90
    while cap.isOpened():
        _, frame = cap.read()
        cannyImg = canny(frame)
        croppedImg = limit(cannyImg)
        lines = cv2.HoughLinesP(croppedImg, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=40)

        if lines is not None:
            avgLines = avgSlopeInt(frame, lines)
            headingImage, currAngle = steering(frame, avgLines, currAngle)
            lineImg = makePoly(frame, avgLines)
            bothImg = cv2.addWeighted(frame, 0.8, lineImg, 1, 1)
            tripleImg = cv2.addWeighted(bothImg, 0.5, headingImage, 1, 1)
            textCenter = cv2.getTextSize(str(currAngle-90), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            arrowCenter = cv2.getTextSize('->', cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            tripleImg = colors(tripleImg, currAngle, textCenter, arrowCenter)
            cv2.imshow(f"Lane Detection: {test} {shp.shape[:-1]} - press 'q' to quit", tripleImg)
        else:
            height, width, _ = frame.shape
            cv2.addWeighted(frame, 0.5, frame, 1, 1)
            textCenter = cv2.getTextSize(str(currAngle-90), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            arrowCenter = cv2.getTextSize('->', cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            doubleImg = cv2.putText(frame, '0', ((shp.shape[1]-textCenter[0]), textCenter[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow(f"Lane Detection: {test} {shp.shape[:-1]} - press 'q' to quit", doubleImg)
        if cv2.waitKey(25) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
