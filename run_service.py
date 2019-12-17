from flask import Flask, request, redirect, jsonify
from model import RFBText
import os
import cv2
import numpy as np
import tensorflow as tf
from TextDetection import textbox_detection
from PageDetection import page_detection
from TextDeskew import text_deskew
from TextRecognition import text_recognition
from TextCorrection import text_correction
from TextParsing import text_parsing
import datetime


model_path = "./model/epoch_258_loss_0.010577003471553326.hdf5"

global rfb_text_model
rfb_text_model = RFBText(model_weights_path=model_path)()
global graph
graph = tf.get_default_graph()
debug_images_path = "./debug_images_service"


app = Flask(__name__)
app.config["IMAGE_UPLOADS"] = "./uploads"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG"]


def allowed_image(filename):

    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True, ext
    else:
        return False, ext


def allowed_image_filesize(filesize):

    if int(filesize) <= app.config["MAX_IMAGE_FILESIZE"]:
        return True
    else:
        return False


@app.route("/upload_image", methods=["POST"])
def upload_image():
    cv_information = {}
    if request.method == "POST":
        if request.files:

            image = request.files["image"]
            if image.filename == "":
                print("No filename")
                return redirect(request.url)

            is_allowed_ext, ext = allowed_image(image.filename)
            if is_allowed_ext:
                datatime = datetime.datetime.now()
                id = str(datatime).replace(".", "").replace(":", "").replace("-", "").replace(" ", "")
                filename = "temp-image_" + id + ". " + ext

                image_path = os.path.join(app.config["IMAGE_UPLOADS"], filename)
                image.save(image_path)

                print("Image saved")

                filename = ".".join(filename.split(".")[:-1])
                image = cv2.imread(image_path)
                image_copy = np.copy(image)

                # Step 1: Text boxes detection
                print('Step 1: text boxes detection')
                predicted_boxes = textbox_detection(rfb_text_model, graph, image_copy, filename, debug_images_path, False)

                # Step 2: Page detection
                print('Step 2: page detection')
                page_image = page_detection(image_copy, predicted_boxes, filename, debug_images_path)

                # Step 3: Text deskew
                print('Step 3: text deskew')
                page_image, angle = text_deskew(page_image, filename, debug_images_path)

                # Step 4: Text recognition
                print('Step 4: text recognition')
                path_to_text = text_recognition(rfb_text_model, graph, page_image, filename, debug_images_path)

                # Step 5: Text correction
                print('Step 5: text correction')
                path_to_text_after_correction = text_correction(path_to_text)


                # Step 6: Text parsing
                print('Step 6: text parsing')
                cv_information = text_parsing(path_to_text_after_correction)

    return jsonify(cv_information)

def run_server():
    app.run(port=8888)


if __name__=="__main__":
    run_server()

# def run(model, input_path, debug_images_path):
#     print('Predicting from test_images at path: {0}'.format(input_path))
#
#     input_path_list = get_path_list(input_path)
#
#     for image_path, filename in input_path_list:
#         # convert: akansa3@uic.edu Aniket Kansara.jpg -> akansa3@uic.edu Aniket Kansara
#         filename = ".".join(filename.split(".")[:-1])
#         image = cv2.imread(image_path)
#         image_copy = np.copy(image)
#
#         # Step 1: Text boxes detection
#         print('prediction in progress')
#         predicted_boxes = textbox_detection(model, image_copy, filename, debug_images_path, False)
#
#         # Step 2: Page detection
#         page_image = page_detection(image_copy, predicted_boxes, filename, debug_images_path)
#
#         # Step 3: Text deskew
#         page_image, angle = text_deskew(page_image, filename, debug_images_path)
#
#         # Step 4: Text recognition
#         path_to_text = text_recognition(model, page_image, filename, debug_images_path)
#         print(path_to_text)
#
#         # Step 5: Text correction
#         path_to_text_after_correction = text_correction(path_to_text)
#
#
#         # Step 6: Text parsing
#         cv_information = text_parsing(path_to_text_after_correction)
#
#
#
# model_path = "./model/epoch_258_loss_0.010577003471553326.hdf5"
# rfb_text_model = RFBText(model_weights_path=model_path)()
# debug_images_path = "./debug_images"
# # input_path = "./cv_images_data/full_data"
# input_path = "./cv_images_data/small_data"
#
# run(model=rfb_text_model, input_path=input_path, debug_images_path=debug_images_path)
