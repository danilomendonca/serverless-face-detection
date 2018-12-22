# import json
import numpy as np
from PIL import Image
import cv2
# import time
import base64
import io

cascade_classifier = cv2.CascadeClassifier("./model/haarcascade_frontalface_alt.xml")
# print "Done preparing face detection classifier!!"

def main(args):
    image = Image.open(io.BytesIO(base64.b64decode(args["image"])))
    # image = Image.open("./testImages/armstrong.jpeg")
    height = image.size[1]

    # PILLOW works in RGB, openCV in BGR
    open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # obtain a gray scale image
    open_cv_gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    # equalize the gray scale image to improve the result
    open_cv_gray_image = cv2.equalizeHist(open_cv_gray_image)

    # compute minimum face size (5% of the frame height)
    absolute_face_height = int(round(height*0.05))

    # start_millis = int(round(time.time() * 1000))
    faces_rects = cascade_classifier.detectMultiScale(open_cv_gray_image,
                                                      scaleFactor=1.2,
                                                      minNeighbors=2,
                                                      minSize=(absolute_face_height, absolute_face_height))
    # draw green rectangles around faces found
    for (x,y,w,h) in faces_rects:
       # image, top left corner and bottom right corner, green in BGR scale, width of rectangle drawn
       cv2.rectangle(open_cv_image, (x,y), (x+w,y+h), (0,255,0), 3)

    # millis = int(round(time.time() * 1000)) - start_millis
    # print "OpenCV Detection and draw time: {} milliseconds".format(millis)

    # reconvert image from BGR to RGB
    result_image = Image.fromarray(cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB))
    # result_image.show()

    # base 64 encode response
    result_buffer = io.BytesIO()
    result_image.save(result_buffer, format='PNG')
    encoded_result = base64.b64encode(result_buffer.getvalue())
    return {"statusCode": 200, "headers": {"Content-Type": "application/json"},"body": {"image": "{}".format(encoded_result)}}
