import cv2
import os
import numpy as np


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def get_outer_line_points(listPoint, h, w):
    ## sort
    listY = sorted(listPoint,key=lambda x: x[1])
    listLinePoint = []
    # top
    leftTop = listY[0]
    rightTop = listY[1]
    for pt in listY:
        if abs(pt[0] - leftTop[0]) > 20:
            topLeft = pt
            break
    listLinePoint.append((leftTop,rightTop))
    #bottom
    listY = sorted(listPoint,key=lambda x: x[1],reverse=True)
    leftBottom = listY[0]
    rightBottom = listY[1]
    for pt in listY:
        if abs(pt[0] - leftBottom[0]) > 20:
            rightBottom = pt
            break
    listLinePoint.append((leftBottom,rightBottom))
    ## sort
    listX = sorted(listPoint,key=lambda x: x[0])
    # left
    # top left
    # (h,w) = image.shape[:2]
    topLeft = listX[0]
    for pt in listX:
        if pt[1] < h//2:
            topLeft = pt
            break
    # revListX = reversed(listX)
    bottomLeft = listX[1]
    for pt in listX:
        if pt[1] > h//2:
            bottomLeft = pt
            break
    listLinePoint.append((topLeft,bottomLeft))
    #right
    # top right
    listX = sorted(listX,key=lambda x: x[0],reverse=True)
    topRight = listX[0]
    for pt in listX:
        if pt[1] < h//2:
            topRight = pt
            break
    # bottom right
    bottomRight = listX[1]
    for pt in listX:
        if pt[1] > h//2:
            bottomRight = pt
            break
    listLinePoint.append((topRight,bottomRight))
    return  listLinePoint


def draw_four_corner_points(image, four_corner_points, size=15):
    for index,pt in enumerate(four_corner_points):
        x = int(pt[0])
        y = int(pt[1])
        four_corner_points[index] = (x, y)
    for index, pt in enumerate(four_corner_points):
        if index < 3:
            cv2.line(image, pt, four_corner_points[index+1], (255, 255, 0), size)
        else:
            cv2.line(image, pt, four_corner_points[0], (255, 255, 0), size)
    for index,pt in enumerate(four_corner_points):
        cv2.circle(image, pt, size, (0, 0, 0), -1)
    return image


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


#
def page_detection(image, predicted_box, filename, debug_path):
    image_copy = np.copy(image)
    image_convex, convex_points = find_convex(image_copy, predicted_box)
    #cv2.imwrite(os.path.join(debug_path, filename+"_2_convext.jpg"), image_convex)

    image_four_corner_points, four_corner_points = find_four_corner_points(image_convex, convex_points)
    #cv2.imwrite(os.path.join(debug_path, filename + "_2_four_corner_points.jpg"), image_four_corner_points)

    four_corner_points = np.asarray(four_corner_points, dtype='float')
    page_image = four_point_transform(image, four_corner_points)
    #cv2.imwrite(os.path.join(debug_path, filename + "_2_page.jpg"), page_image)

    return page_image


def find_convex(image, predicted_box):
    newList = []
    for box in predicted_box:
        for (x, y) in box:
            newList.append([x, y])
    temp = cv2.convexHull(np.array(newList, dtype='float32'))
    convex_points = []
    for index, pt in enumerate(temp):
        cv2.circle(image, tuple(pt[0]), 5, (0, 255, 0), -1)
        convex_points.append(tuple(pt[0]))
        if index == len(temp) - 1:
            cv2.line(image, tuple(pt[0]), tuple(temp[0][0]), 255, 2)
            break
        cv2.line(image, tuple(pt[0]), tuple(temp[index + 1][0]), 255, 2)
    return image, convex_points


def find_four_corner_points(image, convex_points):
    list_convex_points = []
    (h, w) = image.shape[:2]
    for pt in convex_points:
        list_convex_points.append(pt)
    listLinePoint = get_outer_line_points(list_convex_points, h, w)
    pt1 = line_intersection(listLinePoint[0], listLinePoint[2])
    pt2 = line_intersection(listLinePoint[0], listLinePoint[3])
    pt3 = line_intersection(listLinePoint[1], listLinePoint[2])
    pt4 = line_intersection(listLinePoint[1], listLinePoint[3])
    four_corner_points = [pt1, pt2, pt4, pt3]
    image = draw_four_corner_points(image, four_corner_points)

    return image, four_corner_points

