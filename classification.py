from keras.models import load_model
import cv2
import numpy as np

model = load_model("test_model.h5")

def prepImage(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255
    img = np.expand_dims(img,0)
    return img

img = cv2.imread("5.png")
img = prepImage(img)

prediction = model.predict(img)
print(prediction.argmax())
print(prediction[0][5])
img3 = cv2.imread("3.png")
img3 = cv2.resize(img3,(28,28))

img3 = prepImage(img3)
prediction = model.predict(img3)
print(prediction)
print(prediction.argmax())