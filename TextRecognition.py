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


def drawArrow(word1, word2, image, color):
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
    # cv2.imshow("image", image)
    # cv2.waitKey(0)


def drawBox(box, image):
    cv2.polylines(image,
                  [box.astype(np.int32).reshape((-1, 1, 2))],
                  True,
                  color=(0, 0, 0),
                  thickness=2)


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
                drawArrow(word, candidate, image, (0, 0, 255))
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
                drawArrow(word, candidate, image, (0, 255, 0))
                word = candidate
            else:
                break
    return afterTextLines, image, image_copy


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
    doc, textLineImage, textBoxImage = text_lines_detection(
        predicted_boxes, image_copy)
    line_number = 0
    text_file = open(path_to_text, "w+")
    for line in doc:
        line_number += 1
        line_text = ""
        for word_box in line:
            # origin image with box
            # cv2.polylines(image,
            #             [word_box.astype(np.int32).reshape((-1, 1, 2))],
            #             True,
            #             color=(0, 0, 255),
            #             thickness=2)
            word_box_text = four_point_transform(image, word_box)
            config = "-l eng --psm 7 --oem 1"
            text = pytesseract.image_to_string(word_box_text, config=config)
            text = text.replace("\n", "")
            line_text = line_text + " " + text
        text_file.write(line_text + "\n")
    text_file.close()
    return image, path_to_text, textLineImage, textBoxImage

# this function for text recognize
def text_recognition(model, graph, page_image, filename, debug_images_path):
    predicted_boxes = textbox_detection(
        model, graph, page_image, filename, debug_images_path, True)
    image, path_to_text, textLineImage, textBoxImage = process_text_boxes_in_image(
        page_image, predicted_boxes, os.path.join(debug_images_path, filename + "_4_recognition.txt"))
    cv2.imwrite(os.path.join(debug_images_path, filename +
                             "4_text_lines.jpg"), textLineImage)
    # cv2.imwrite(os.path.join(debug_images_path,filename+"4_text_boxes.jpg"),textBoxImage)
    return path_to_text
