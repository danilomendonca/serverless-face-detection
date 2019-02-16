import numpy as np
import cv2
import base64

cascade_classifier = cv2.CascadeClassifier("./model/haarcascade_frontalface_alt.xml")

def main(args):
    image = cv2.imdecode(np.fromstring(base64.b64decode(args["image"]), dtype=np.uint8),1)
    #image = cv2.imread("./testImages/armstrong.jpeg")
    height = image.shape[:2][0]

    # Android works in RGB, openCV in BGR
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

    # reconvert image from BGR to RGB
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
    # https://docs.opencv.org/3.0-beta/modules/imgcodecs/doc/reading_and_writing_images.html#imencode
    #encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
    #retval, buffer = cv2.imencode('.jpg', open_cv_image, encode_param)
    retval, buffer = cv2.imencode('.png', open_cv_image)
    return {"statusCode":200,"headers":{"Content-Type":"application/json"},"body":{"image":base64.b64encode(buffer)}}


#if __name__=="__main__":
#    main({})
