import cv2
from keras.models import load_model
import numpy as np
def prepImage(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255
    img = np.expand_dims(img,0)
    return img
count = 0
model = load_model("test_model.h5")
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        resized_frame = cv2.resize(frame,(28,28))
        resized_frame = prepImage(resized_frame)
        prediction = model.predict(resized_frame)
        cv2.putText(frame,str(prediction.argmax()) + "    " + str(prediction[0][prediction.argmax()]),(50,50),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0),1)
        count += 1
        cv2.imshow("frame",frame)
        if cv2.waitKey(1) &0xFF == ord("q"):
            break
    else: break
cap.release()
cv2.destroyAllWindows()