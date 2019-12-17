import cv2
import numpy as np
import east_utils
from keras.applications.resnet50 import preprocess_input
import os


def paint_text_boxes_to_image(image, predicted_boxes):
    """
    Draw the quad-boxes on-to the image
    :param image:
    :param predicted_boxes:
    :return:
    """
    for box in predicted_boxes:
        cv2.polylines(image,
                      [box.astype(np.int32).reshape((-1, 1, 2))],
                      True,
                      color=(0, 0, 255),
                      thickness=2)

    return image


def textbox_detection(model, graph, image, filename, debug_path, recognition_mode = False):
    """
    Predict the bounding box quads for a single image
    :param model:
    :param image:
    :param debug_path: to save images for debugging
    :return:
    """
    image_copy = np.copy(image)
    image_copy_after_filter = np.copy(image)
    im_resized, (ratio_h, ratio_w) = east_utils.resize_image(image)
    im_resized = cv2.cvtColor(im_resized, cv2.COLOR_BGR2RGB)

    image_batch = np.expand_dims(im_resized, axis=0)

    image_batch = preprocess_input(image_batch)

    with graph.as_default():
        preds = model.predict(image_batch)
        score, geometry = preds

    boxes = east_utils.detect(score_map=score,
                              geo_map=geometry)

    predicted_boxes = list()
    if boxes is not None:
        boxes = boxes[:, :8].reshape((-1, 4, 2))
        boxes[:, :, 0] /= ratio_w
        boxes[:, :, 1] /= ratio_h

        for box in boxes:
            box = east_utils.sort_poly(box.astype(np.int32))

            # reduce false positives
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue

            predicted_boxes.append(box)

    print('prediction done')
    # if not recognition_mode:
    #     cv2.imwrite(os.path.join(debug_path, filename+"_0_src.jpg"), image)

    # pred_image = paint_text_boxes_to_image(image_copy, predicted_boxes)

    # if not recognition_mode:
    #     cv2.imwrite(os.path.join(debug_path, filename+"_1_textbox.jpg"), pred_image)
    # else:
    #     cv2.imwrite(os.path.join(debug_path, filename + "_4_textbox_recognition.jpg"), pred_image)

    # predicted_boxes = outlier_filter(predicted_boxes)
    # pred_image_after_filter = paint_text_boxes_to_image(image_copy_after_filter, predicted_boxes)

    # if not recognition_mode:
    #     cv2.imwrite(os.path.join(debug_path, filename + "_1_textbox_after_filter.jpg"), pred_image_after_filter)

    return predicted_boxes


def outlier_filter(predicted_boxes):
    predicted_boxes_after_filter = predicted_boxes
    return predicted_boxes_after_filter

