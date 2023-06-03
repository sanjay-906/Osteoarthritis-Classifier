# Write your code here :-)
import numpy as np
import cv2
import sklearn
from keras.models import load_model
import streamlit as sl
import keras
from io import BytesIO, StringIO
from PIL import Image, ImageOps
import tensorflow as tf
import urllib.request


class_names=['Present', 'Absent']

def compute(path):

    model=load_model('model1.h5', compile=False)
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(3e-4),
              metrics=['accuracy'])
    x=model.predict(path)

    score = tf.nn.softmax(x[0])
    sl.write(x)
    sl.write(score)
    output="There is {:.2f}% chance that Osteoarthritis is {} in the given Xray Image.".format( 100 * np.max(score), class_names[np.argmax(score)])
    sl.write(output)




def main():

    sl.title('Osteoarthritis Detection')
    file= sl.file_uploader("Choose a file",type = ['jpg', 'png','jpeg'])

    if file is None:
        sl.text("Please upload an image file")
        sl.text("No Image? Download few:")
        link = '[images](https://drive.google.com/drive/folders/1Yjd-Fv5n2Nb6Rl3zrGNuVymtivLn2749?usp=sharing)'
        sl.markdown(link, unsafe_allow_html=True)
    else:
        img = cv2.imdecode(np.fromstring(file.read(), np.uint8), 1)
        sl.image(img)
        new_array = cv2.resize(img, (200, 200))
        X=np.array(new_array).reshape(-1,200,200,1)
        X=X/255.0
        compute(X)




if __name__ == '__main__':
    main()
