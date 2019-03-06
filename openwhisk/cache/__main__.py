import numpy as np
import cv2
import base64

def main(args):
    global cache
    if 'cache' not in globals():
        cascade_classifier = cv2.CascadeClassifier("./model/haarcascade_frontalface_alt.xml")
        cache = cascade_classifier
    else:
        cascade_classifier = cache

    image = cv2.imdecode(np.fromstring(base64.b64decode(args["image"]), dtype=np.uint8),1)
    open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    open_cv_gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    open_cv_gray_image = cv2.equalizeHist(open_cv_gray_image)
    absolute_face_height = int(round(image.shape[:2][0]*0.05))
    faces_rects = cascade_classifier.detectMultiScale(open_cv_gray_image,
                                                      scaleFactor=1.2,
                                                      minNeighbors=2,
                                                      minSize=(absolute_face_height, absolute_face_height))
    for (x,y,w,h) in faces_rects:
       cv2.rectangle(open_cv_image, (x,y), (x+w,y+h), (0,255,0), 3)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
    #retval, buffer = cv2.imencode('.png', open_cv_image)
    retval, buffer = cv2.imencode('.jpg', open_cv_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return {"image": base64.b64encode(buffer),
            "cache": cache}


#if __name__=="__main__":
#    main({})
