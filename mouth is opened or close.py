#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import pandas as pd
import numpy as np
import tensorflow as tf


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, InputLayer, Dropout, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping as ES
from IPython.display import clear_output as cls


import plotly.express as px
import matplotlib.pyplot as plt


# In[6]:


batchSize = 32
imageShape = (224, 224)
epochs = 10
channels = 3


# In[11]:


train_dir = "C:/Users/vinayak khandelwal/Downloads/mouth/train/"
val_dir = "C:/Users/vinayak khandelwal/Downloads/mouth/test/"


# In[12]:


class_names = sorted(os.listdir(train_dir))
class_names


# In[13]:


class_dis = [len(os.listdir(train_dir + name)) for name in class_names]
class_dis


# In[15]:


train_gen = ImageDataGenerator(rescale=(1./255.), validation_split=0.2)
val_gen = ImageDataGenerator(rescale=(1./255.))


# In[16]:


imageShape = (224, 224)

train_ds = train_gen.flow_from_directory(train_dir,target_size=imageShape,batch_size=batchSize, subset='training', class_mode='categorical')
val_ds = train_gen.flow_from_directory(train_dir,target_size=imageShape,batch_size=batchSize, subset='validation', class_mode='categorical')


# In[17]:


test_ds = val_gen.flow_from_directory(val_dir,target_size=imageShape,batch_size=batchSize, class_mode='categorical')


# In[18]:


def plot_images(data, class_names):
    
    r, c = 3, 4
    imgLen = r*c
    
    plt.figure(figsize=(20, 15))
    i = 1
    
    for images, labels in iter(data):
        
        
        id = np.random.randint(len(images))
#         img = images[id].numpy().astype('uint8')
        img = tf.expand_dims(images[id], axis=0)
        lab = class_names[np.argmax(labels[id])]
        
        plt.subplot(r, c, i)
        plt.imshow(img[0])
        plt.title(lab)
        plt.axis('off')
        cls()
        
        i+=1
        if i > imgLen:
            break
    plt.show()


# In[19]:


plot_images(train_ds, class_names)


# In[20]:


def get_model():
    
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    for layers in base_model.layers:
        layers.trainable = False
        
    base_model_output = base_model.output
    
    x = Flatten()(base_model_output)
    x = Dense(512, activation='relu')(x)
    x = Dense(len(class_names), activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=x)
    return model


# In[21]:


model = get_model()

model.compile(loss="categorical_crossentropy",
             optimizer="sgd",
             metrics=["accuracy"])

model.summary()


# In[25]:


history = model.fit(train_ds,
                   validation_data=val_ds,
                   epochs=10,
                   steps_per_epoch=len(train_ds),
                   validation_steps=len(val_ds),
                   callbacks=[ES(monitor="val_loss", patience=5)])


# In[26]:


model.save('mouth_model.h5')


# In[27]:


def predictImages(data, class_names, model):
    
    r, c = 3, 4
    imgLen = r*c
    plt.figure(figsize=(20, 15))
    i = 1
    
    for images, labels in iter(data):
        
        id = np.random.randint(len(images))
        img = tf.expand_dims(images[id], axis=0)
        
        plt.subplot(r, c, i)
        plt.imshow(img[0])
        
        predicted = model.predict(img)
        predicted = class_names[np.argmax(predicted)]
        actual = class_names[np.argmax(labels[id])]
        
        plt.title(f"Actual: {actual}\nPredicted: {predicted}")
        plt.axis('off')
        cls()
        
        i+=1
        if i > imgLen:
            break
            
    plt.show()


# In[28]:


predictImages(test_ds, class_names, model)


# In[1]:


import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained model
model = load_model('mouth_model.h5')

# Define image size for model input
img_size = (224, 224)

# Map class indices to class names
class_names = {0: "close", 1: "open"}

# GUI setup
class ImageClassifierApp:
    def __init__(self, master):
        self.master = master
        self.master.title(" Mouth Image Classifier")

        # Create widgets
        self.label = tk.Label(self.master, text="Select an image:")
        self.label.pack()

        self.browse_button = tk.Button(self.master, text="Browse", command=self.browse_image)
        self.browse_button.pack()

        self.classify_button = tk.Button(self.master, text="Classify", command=self.classify_image)
        self.classify_button.pack()

        self.result_label = tk.Label(self.master, text="")
        self.result_label.pack()

        self.image_label = tk.Label(self.master)
        self.image_label.pack()

        # Initialize img_tk as a class attribute
        self.img_tk = None

    def browse_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.display_image(file_path)

    def classify_image(self):
        if hasattr(self, 'image_np'):
            # Preprocess the image
            img = Image.fromarray(self.image_np)
            img = img.resize(img_size)
            img_array = np.expand_dims(np.array(img), axis=0) / 255.0

            # Perform image classification
            predictions = model.predict(img_array)
            class_index = np.argmax(predictions)
            class_name = class_names.get(class_index, "Unknown")

            # Display result
            self.result_label.config(text=f"Prediction: {class_name}")
        else:
            self.result_label.config(text="No image selected")

    def display_image(self, file_path):
        img = Image.open(file_path)
        img = img.resize((300, 300))
        self.img_tk = ImageTk.PhotoImage(img)

        self.image_label.config(image=self.img_tk)

        # Save numpy image for later classification
        self.image_np = np.array(img)

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()


# In[ ]:





# In[ ]:




