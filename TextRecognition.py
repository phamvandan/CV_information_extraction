from TextDetection import textbox_detection
from PageDetection import order_points, four_point_transform
import numpy as np
import cv2
import pytesseract
import os


def caculate_center_box(box):
    temp = box[:, 1]
    return np.mean(temp)


def isSameLine(box1, listBox):
    mean = 0
    for box in listBox:
        mean = mean + caculate_center_box(box)
    mean = mean / len(listBox)
    y1 = caculate_center_box(box1)
    # y2 = caculateCenterY(box2)
    if abs(y1-mean) <= 20:
        return True
    return False

import numpy as np
def drawArrow(word1, word2, image, color,mask):
    (x11, y11) = (word1[0][0], word1[0][1])
    (x12, y12) = (word1[2][0], word1[2][1])
    (x21, y21) = (word2[0][0], word2[0][1])
    (x22, y22) = (word2[2][0], word2[2][1])
    pt1 = ((x11+x12)//2, (y11+y12)//2)
    pt2 = ((x21+x22)//2, (y21+y22)//2)
    cv2.polylines(image,
                  [word1.astype(np.int32).reshape((-1, 1, 2))],
                  True,
                  color,
                  thickness=1)
    cv2.polylines(image,
                  [word2.astype(np.int32).reshape((-1, 1, 2))],
                  True,
                  color,
                  thickness=1)
    cv2.arrowedLine(image, (x11, y11), (x21, y21), color, 2)
    (x11, y11) = (word1[1][0], word1[1][1])
    (x21, y21) = (word2[1][0], word2[1][1])
    cv2.line(mask,(x11, y11), (x21, y21),255,2)
    (x41, y41) = (word1[2][0], word1[2][1])
    (x31, y31) = (word2[2][0], word2[2][1])
    # cv2.line(mask,(x11, y11), (x21, y21),255,2)
    # pts = np.array([(x11,y11),(x21,y21),(x31,y31),(x41,y41)],dtype='int')
    # cv2.drawContours(mask,[pts],0,255,-1)
    # cv2.imshow("image", image)
    # cv2.waitKey(0)


def drawBox(box, image):
    cv2.polylines(image,
                  [box.astype(np.int32).reshape((-1, 1, 2))],
                  True,
                  color=(0, 0, 0),
                  thickness=2)

def drawRectangle(box,image):
    xmin,ymin,xmax,ymax = box
    cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(0,0,0),2)

def yCondition(word1, word2, heightMean):
    (x11, y11) = (word1[0][0], word1[0][1])
    (x12, y12) = (word1[2][0], word1[2][1])
    (x21, y21) = (word2[0][0], word2[0][1])
    (x22, y22) = (word2[2][0], word2[2][1])
    y11 = y11 + heightMean//8
    y12 = y12 - heightMean//8
    y21 = y21 + heightMean//8
    y22 = y22 - heightMean//8
    if (y11 <= y21 and y12 >= y21) or (y11 <= y22 and y21 >= y22):
        return True
    if (y21 <= y11 and y22 >= y11) or (y21 <= y12 and y22 >= y12):
        return True 
    return False


def caculateAvergeHeight(predicted_boxes):
    height = 0
    for box in predicted_boxes:
        height = height + (box[3][1] - box[0][1] + box[2][1] - box[1][1])/2
    if len(predicted_boxes) == 0:
        return 0
    return height/len(predicted_boxes)


def caculateAvergeWidth(predicted_boxes):
    width = 0
    for box in predicted_boxes:
        width = width + (box[1][0] - box[0][0] + box[2][0] - box[3][0])/2
    if len(predicted_boxes) == 0:
        return 0
    return width/len(predicted_boxes)

import math
def isGoodAngle(pt1,pt2):
    (xl,yl) = (pt1[0],pt1[1])
    (xr,yr) = (pt2[0],pt2[1])
    angle = math.degrees(math.atan(float(abs(float(yr-yl)/float(xr-xl)))))
    if angle <15:
        return True
    return False

def text_lines_detection(predicted_boxes, image):
    mask = np.zeros(image.shape[:2])
    preTextLines = []
    afterTextLines = []
    widthMean = caculateAvergeWidth(predicted_boxes)
    heightMean = caculateAvergeHeight(predicted_boxes)
    print("averge width: "+str(widthMean))
    print("averge height: "+str(heightMean))
    image_copy = image.copy()
    for box in predicted_boxes:
        preTextLines.append(box)
        drawBox(box, image_copy)
    lineNumber = -1
    # sort follow y coordinate
    preTextLines = sorted(preTextLines, key=lambda x: x[0][1])
    while len(preTextLines) > 0:
        # choose candidate
        word = preTextLines.pop(0)
        drawBox(word, image)
        temp = word
        afterTextLines.append([word])
        lineNumber = lineNumber + 1
        # find right candidates
        while True:
            candidates = []
            for index, candidate in enumerate(preTextLines):
                if candidate[0][0] > word[0][0] and isGoodAngle(word[0],candidate[0]) and yCondition(candidate, word, heightMean):
                    candidates.append((index, candidate))
            minDistace = 10000
            minCandidateIndex = None
            for index, candidate in candidates:
                distance = candidate[0][0] - word[2][0]
                if distance >= 2*widthMean:
                    continue
                if distance < minDistace:
                    minCandidateIndex = index
                    minDistace = distance
            if minCandidateIndex is not None:
                candidate = preTextLines.pop(minCandidateIndex)
                afterTextLines[lineNumber].append(candidate)
                drawArrow(word, candidate, image, (0, 0, 255),mask)
                word = candidate
            else:
                break
        # find left candidates
        word = temp
        while True:
            candidates = []
            for index, candidate in enumerate(preTextLines):
                if word[0][0] > candidate[0][0] and isGoodAngle(word[0],candidate[0]) and yCondition(candidate, word, heightMean):
                    candidates.append((index, candidate))
            minDistace = 10000
            minCandidateIndex = None
            for index, candidate in candidates:
                distance = word[0][0] - candidate[2][0]
                if distance >= 2*widthMean:
                    continue
                if distance < minDistace:
                    minCandidateIndex = index
                    minDistace = distance
            if minCandidateIndex is not None:
                candidate = preTextLines.pop(minCandidateIndex)
                afterTextLines[lineNumber].insert(0, candidate)
                drawArrow(word, candidate, image, (0, 255, 0),mask)
                word = candidate
            else:
                break
    return afterTextLines, image, image_copy

def assignCoordinate(entireBox,newBox):
    xmin,ymin,xmax,ymax = entireBox
    x1,y1 = newBox[0]
    x2,y2 = newBox[1]
    x3,y3 = newBox[2]
    x4,y4 = newBox[3]
    if x1<xmin:
        xmin = x1
    elif x4<xmin:
        xmin = x4
    if y1<ymin:
        ymin = y1 
    elif y2<ymin:
        ymin = y2
    if x2>xmax:
        xmax = x2
    elif x4>xmax:
        xmax = x4
    if y3 > ymax:
        ymax = y3
    elif y4>ymax:
        ymax = y4
    entireBox = (xmin,ymin,xmax,ymax)
    return entireBox

def extendEntireBox(entireBox,width,height,ratio):
    xmin,ymin,xmax,ymax = entireBox
    print("DELTA")
    deltaX= width//ratio
    xmin = int(max(xmin - deltaX,0))
    xmax = int(min(xmax + deltaX,width))
    entireBox = (xmin,ymin,xmax,ymax)
    print(str(deltaX))
    return entireBox

import  math
def caculateAngleOfBox(pt1,pt2):
    x1,y1 = pt1
    x2,y2 = pt2
    if x1>=x2:
        return None
    angle = math.atan(float(float(y1-y2)/float(abs(x1-x2))))
    # print(angle)
    return angle

def standardPoint(pt,h,w):
    x,y = pt
    if x<=0:
        x = 0
    elif x >= w:
        x = w
    if y<= 0:
        y=0
    elif y>=h:
        y=h
    pt = (round(x),round(y))
    return pt

def extendLine(pt1,pt2,delta,w,h):
    swap = False
    if pt1[0] > pt2[0]:
        temp = pt1
        pt1 = pt2
        pt2 = temp
        swap = True
    pt1 = standardPoint(pt1,h,w)
    pt2 = standardPoint(pt2,h,w)
    x1,y1 = pt1
    x2,y2 = pt2
    angle = caculateAngleOfBox(pt1,pt2)
    x_1 = x1
    y_1 = y1
    x_2 = x2
    y_2 = y2
    if angle is None:
        return pt1,pt2
    else:
        leftAngle = angle
        rightAngle = 0 - angle
        x_1 = round(x1 - delta)
        y_1 = round(y1 + delta*math.tan(leftAngle))
        if x_1 <0 or x_1>w or y_1<0 or y_1>h:
            (x_1,y_1) = standardPoint((x_1,y_1),h,w)
        x_2 = round(x2 + delta)
        y_2 = round(y2 + delta*math.tan(rightAngle))
        if x_2 <0 or x_2>w or y_2<0 or y_2>h:
            (x_2,y_2) = standardPoint((x_2,y_2),h,w)
    expt1 = (x_1,y_1)
    expt2 = (x_2,y_2)
    if swap:
        temp = expt1
        expt1 = expt2
        expt2 = temp
    return expt1,expt2

def printBox(box):
    print(box)
    print(type(box))

# def drawBox(box,img):
#     cv2.drawContours(img, [box], 0, 255, 2)

def Swap(a,b):
    c = a
    a = b
    b = c

def processBox(box):
    box = sorted(box,key=lambda x: x[0])
    pt1 = None
    pt2 = None
    pt3 = None 
    pt4 = None
    if box[0][1] > box[1][1]:
        pt1 = box[1]
        pt4 = box[0]
    else:
        pt1 = box[0]
        pt4 = box[1]
    if box[2][1] > box[3][1]:
        pt2 = box[3]
        pt3 = box[2]
    else:
        pt2 = box[2]
        pt3 = box[3]
    box = np.array([pt1,pt2,pt3,pt4],dtype='int')
    return box

def text_lines_detection_version2(predicted_boxes, image):
    mask = image.shape[:2]
    (h,w) = image.shape[:2]
    preTextLines = []
    afterTextLines = []
    afterTextEntireLines = []
    widthMean = caculateAvergeWidth(predicted_boxes)
    heightMean = caculateAvergeHeight(predicted_boxes)
    print("averge width: "+str(widthMean))
    print("averge height: "+str(heightMean))
    image_copy = image.copy()
    for box in predicted_boxes:
        preTextLines.append(box)
        drawBox(box, image_copy)
    lineNumber = -1
    # sort follow y coordinate
    preTextLines = sorted(preTextLines, key=lambda x: x[0][1])
    while len(preTextLines) > 0:
        # choose candidate
        word = preTextLines.pop(0)
        drawBox(word, image)
        temp = word
        afterTextLines.append([word])
        lineNumber = lineNumber + 1
        # assign
        xmin = word[0][0]
        ymin = word[0][1]
        xmax = word[1][0]
        ymax = word[1][1]
        entireBox = (xmin,ymin,xmax,ymax)
        entireBox = assignCoordinate(entireBox,word)
        # find right candidates
        while True:
            candidates = []
            for index, candidate in enumerate(preTextLines):
                if candidate[0][0] > word[0][0] and isGoodAngle(word[0],candidate[0]) and yCondition(candidate, word, heightMean):
                    candidates.append((index, candidate))
            minDistace = 10000
            minCandidateIndex = None
            for index, candidate in candidates:
                distance = candidate[0][0] - word[2][0]
                if distance >= 2*widthMean:
                    continue
                if distance < minDistace:
                    minCandidateIndex = index
                    minDistace = distance
            if minCandidateIndex is not None:
                candidate = preTextLines.pop(minCandidateIndex)
                afterTextLines[lineNumber].append(candidate)
                entireBox = assignCoordinate(entireBox,candidate)
                drawArrow(word, candidate, image, (0, 0, 255),mask)
                word = candidate
            else:
                break
        # find left candidates
        word = temp
        while True:
            candidates = []
            for index, candidate in enumerate(preTextLines):
                if word[0][0] > candidate[0][0] and isGoodAngle(word[0],candidate[0]) and yCondition(candidate, word, heightMean):
                    candidates.append((index, candidate))
            minDistace = 10000
            minCandidateIndex = None
            for index, candidate in candidates:
                distance = word[0][0] - candidate[2][0]
                if distance >= 2*widthMean:
                    continue
                if distance < minDistace:
                    minCandidateIndex = index
                    minDistace = distance
            if minCandidateIndex is not None:
                candidate = preTextLines.pop(minCandidateIndex)
                afterTextLines[lineNumber].insert(0, candidate)
                entireBox = assignCoordinate(entireBox,candidate)
                drawArrow(word, candidate, image, (0, 255, 0),mask)
                word = candidate
            else:
                break
        entireBox = extendEntireBox(entireBox,w,h,55)
        afterTextEntireLines.append(entireBox)
    return afterTextLines, image, image_copy,afterTextEntireLines

def text_lines_detection_version3(predicted_boxes, image):
    (h,w) = image.shape[:2]
    mask = np.zeros(image.shape[:2])
    preTextLines = []
    afterTextLines = []
    afterTextEntireLines = []
    widthMean = caculateAvergeWidth(predicted_boxes)
    heightMean = caculateAvergeHeight(predicted_boxes)
    print("averge width: "+str(widthMean))
    print("averge height: "+str(heightMean))
    image_copy = image.copy()
    for box in predicted_boxes:
        preTextLines.append(box)
        # drawBox(box, image_copy)
    lineNumber = -1
    # sort follow y coordinate
    preTextLines = sorted(preTextLines, key=lambda x: x[0][1])
    while len(preTextLines) > 0:
        # choose candidate
        word = preTextLines.pop(0)
        drawBox(word, image)
        temp = word
        afterTextLines.append([word])
        lineNumber = lineNumber + 1
        #draw
        # cv2.line(mask,(word[0][0],word[0][1]),(word[1][0],word[1][1]),255,2)
        # cv2.line(mask,(word[2][0],word[2][1]),(word[3][0],word[3][1]),255,2)
        # pts = np.array([(word[0][0],word[0][1]),(word[1][0],word[1][1]),(word[2][0],word[2][1]),(word[3][0],word[3][1])],dtype='int')
        # cv2.drawContours(mask,[pts],0,255,-1)
        # find right candidates
        while True:
            candidates = []
            for index, candidate in enumerate(preTextLines):
                if candidate[0][0] > word[0][0] and isGoodAngle(word[0],candidate[0]) and yCondition(candidate, word, heightMean):
                    candidates.append((index, candidate))
            minDistace = 10000
            minCandidateIndex = None
            for index, candidate in candidates:
                distance = candidate[0][0] - word[2][0]
                if distance >= 2*widthMean:
                    continue
                if distance < minDistace:
                    minCandidateIndex = index
                    minDistace = distance
            if minCandidateIndex is not None:
                candidate = preTextLines.pop(minCandidateIndex)
                afterTextLines[lineNumber].append(candidate)
                drawArrow(word, candidate, image, (0, 0, 255),mask)
                word = candidate
            else:
                break
        # find left candidates
        word = temp
        while True:
            candidates = []
            for index, candidate in enumerate(preTextLines):
                if word[0][0] > candidate[0][0] and isGoodAngle(word[0],candidate[0]) and yCondition(candidate, word, heightMean):
                    candidates.append((index, candidate))
            minDistace = 10000
            minCandidateIndex = None
            for index, candidate in candidates:
                distance = word[0][0] - candidate[2][0]
                if distance >= 2*widthMean:
                    continue
                if distance < minDistace:
                    minCandidateIndex = index
                    minDistace = distance
            if minCandidateIndex is not None:
                candidate = preTextLines.pop(minCandidateIndex)
                afterTextLines[lineNumber].insert(0, candidate)
                drawArrow(word, candidate, image, (0, 255, 0),mask)
                word = candidate
            else:
                break
        # start_word = afterTextLines[lineNumber][0]
        # end_word = afterTextLines[lineNumber][len(afterTextLines[lineNumber])-1]
        # cv2.line(mask,(start_word[0][0],start_word[0][1]),(start_word[3][0],start_word[3][1]),255,2)
        # cv2.line(mask,(start_word[0][0],start_word[0][1]),(start_word[1][0],start_word[1][1]),255,2)
        # cv2.line(mask,(start_word[3][0],start_word[3][1]),(start_word[2][0],start_word[2][1]),255,2)
        # cv2.line(mask,(end_word[1][0],end_word[1][1]),(end_word[2][0],end_word[2][1]),255,2)
        listPoint = []
        for word in afterTextLines[lineNumber]:
            for point in word:
                listPoint.append(point)
        listPoint = np.array(listPoint,dtype='int')
        box = cv2.minAreaRect(listPoint)
        box = cv2.boxPoints(box)
        box = np.int0(box)
        cv2.drawContours(mask, [box], 0, 255, 2)
        cv2.drawContours(image_copy, [box], 0, (0,0,0), 2)
        box = processBox(box)
        extendDelta = abs(box[1][1]-box[2][1])
        print(extendDelta)
        box[0],box[1] = extendLine(box[0],box[1],extendDelta,w,h)
        box[2],box[3] = extendLine(box[2],box[3],extendDelta,w,h)
        cv2.drawContours(mask, [box], 0, 255, 2)
        cv2.drawContours(image_copy, [box], 0, (255,0,0), 2)
        afterTextEntireLines.append(box)
    # mask = cv2.resize(mask,(800,800))
    # cv2.imshow("ok",mask)
    # cv2.waitKey(0)
    return afterTextLines, image, image_copy,mask,afterTextEntireLines

def process_text_boxes_in_image(image, predicted_boxes, path_to_text):
    """
    Draw the quad-boxes on-to the image
    Create mask for text regions
    :param image:
    :param predicted_boxes:
    :param path_to_text: path to save text file after recognize
    :return:
    """
    image_copy = image.copy()
    for index, box in enumerate(predicted_boxes):
        predicted_boxes[index] = order_points(box)
    doc, textLineImage, textBoxImage,mask,afterTextEntireLines = text_lines_detection_version3(
        predicted_boxes, image_copy)
    # for box in afterTextEntireLines:
        # drawRectangle(box,textLineImage)
    line_number = 0
    text_file = open(path_to_text, "w+")
    # # for line in doc:
    # #     line_number += 1
    # line_text = ""
    for word_box in afterTextEntireLines:
        # origin image with box
        # cv2.polylines(image,
        #             [word_box.astype(np.int32).reshape((-1, 1, 2))],
        #             True,
        #             color=(0, 0, 255),
        #             thickness=2)
        word_box_text = four_point_transform(image, word_box)
        config = "-l eng --psm 7 --oem 1"
        line_text = pytesseract.image_to_string(word_box_text, config=config)
        # print(line_text)
        # cv2.imshow("ok",word_box_text)
        # cv2.waitKey(0)
        # text = text.replace("\n", "")
        line_text = line_text + "\n"
        text_file.write(line_text)
    text_file.close()
    return image, path_to_text, textLineImage, textBoxImage,mask

# this function for text recognize
def text_recognition(model, graph, page_image, filename, debug_images_path):
    predicted_boxes = textbox_detection(
        model, graph, page_image, filename, debug_images_path, True)
    image, path_to_text, textLineImage, textBoxImage,mask = process_text_boxes_in_image(
        page_image, predicted_boxes, os.path.join(debug_images_path, filename + "_4_recognition.txt"))
    cv2.imwrite(os.path.join(debug_images_path, filename +
                             "_4_text_lines.jpg"), textLineImage)
    cv2.imwrite(os.path.join(debug_images_path, filename +
                             "_4_text_mask.jpg"), mask)
    cv2.imwrite(os.path.join(debug_images_path, filename +
                             "_4_text_box.jpg"), textBoxImage)
    # cv2.imwrite(os.path.join(debug_images_path,filename+"4_text_boxes.jpg"),textBoxImage)
    return path_to_text
