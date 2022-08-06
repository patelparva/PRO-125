import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time

X=np.load('image.npz')['arr_0']
y=pd.read_csv('labels.csv')['labels']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=52,train_size=0.75,test_size=0.25)

clf=LogisticRegression(solver='saga',multi_class='multinomial').fit(X_train,y_train)

def get_prediction(image):
    im_pil=Image.open(image)

    image_bw=im_pil.convert('L')
    image_bw_resized=image_bw.resize((22,30),Image.ANTIALIAS)

    pixel_filter = 20
    min_pixel = np.percentile(image_bw_resized,pixel_filter)
    image_bw_resized_scaled=np.clip(image_bw_resized-min_pixel,0,255)

    max_pixel=np.max(image_bw_resized)
        
    image_bw_resized_scaled=np.asarray(image_bw_resized_scaled)/max_pixel
    test_sample=np.array(image_bw_resized_scaled).reshape(1,660)
    test_pred=clf.predict(test_sample)

    return test_pred[0]