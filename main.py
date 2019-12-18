from model import RFBText
import os
import cv2
import numpy as np
from TextDetection import textbox_detection
from PageDetection import page_detection
from TextDeskew import text_deskew
from TextRecognition import text_recognition
from TextCorrection import text_correction
from TextParsing import text_parsing
import tensorflow as tf


def get_path_list(root_path):
    filename_list = list()
    for root, _, filenames in os.walk(root_path):
        filename_list.extend([(os.path.join(root, filename), filename) for filename in filenames
                              if filename.lower().endswith(('jpeg', 'png', 'bmp', 'jpg'))
                              ])
    return filename_list


def run(model, input_path, debug_images_path):
    print('Predicting from test_images at path: {0}'.format(input_path))

    input_path_list = get_path_list(input_path)
    # save the debug_images_path
    savePath = debug_images_path
    for image_path, filename in input_path_list:
        # convert: akansa3@uic.edu Aniket Kansara.jpg -> akansa3@uic.edu Aniket Kansara
        filename = ".".join(filename.split(".")[:-1])
        # make new folder in the debug_images_path name akansa3@uic.edu Aniket Kansara
        folderName = filename
        debug_images_path = savePath
        debug_images_path = os.path.join(debug_images_path,folderName)
        if not os.path.isdir(debug_images_path):
            os.mkdir(debug_images_path)
        image = cv2.imread(image_path)
        image_copy = np.copy(image)

        # Step 1: Text boxes detection
        print('prediction in progress')
        predicted_boxes = textbox_detection(rfb_text_model, graph, image_copy, filename, debug_images_path, False)

        # Step 2: Page detection
        page_image = page_detection(image_copy, predicted_boxes, filename, debug_images_path)

        # Step 3: Text deskew
        page_image, angle = text_deskew(page_image, filename, debug_images_path)

        # # Step 4: Text recognition
        path_to_text = text_recognition(model, graph, page_image, filename, debug_images_path)
        print(path_to_text)

        # # Step 5: Text correction
        # path_to_text_after_correction = text_correction(path_to_text)


        # # Step 6: Text parsing
        # cv_information = text_parsing(path_to_text_after_correction)



model_path = "./model/epoch_258_loss_0.010577003471553326.hdf5"
global rfb_text_model
rfb_text_model = RFBText(model_weights_path=model_path)()
global graph
graph = tf.get_default_graph()
debug_images_path = "./debug_images"


input_path = "./cv_images_data/full_data"
# input_path = "./cv_images_data/small_data"

run(model=rfb_text_model, input_path=input_path, debug_images_path=debug_images_path)
