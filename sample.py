from numpy.core.fromnumeric import shape
import streamlit as st
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf

model = tf.keras.models.load_model('Model_1.h5')
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
money_type = ['1000', '10000', '100000', '2000', '20000', '200000', '5000', '50000', '500000']
  

menu = ['camera', 'predict money']
choice = st.sidebar.selectbox('Menu', menu)

if choice == 'predict money':
    image_upload = st.file_uploader('upload file', type = ['jpg', 'png', 'jpeg'])
    if image_upload != None:
        image_np = np.asarray(bytearray(image_upload.read()),dtype = np.uint8)
        img = cv2.imdecode(image_np,1)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224,224))
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        index = np.argmax(prediction[0])
        money = money_type[index]
        st.image(image_upload)
        st.write('This is:', money)
        
        



elif choice == 'camera':
    cam = cv2.VideoCapture(0) # device 0. If not work, try with 1 or 2
    run = st.checkbox('Show webcam')
    capture_button = st.checkbox('Capture')

    captured_image = np.array(None)

    if not cam.isOpened():
        raise IOError("Cannot open webcam")

    FRAME_WINDOW = st.image([])
    while True:
        ret, frame = cam.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        FRAME_WINDOW.image(frame)
        
        # cv2.imshow('My App!', frame)

        # key = cv2.waitKey(1) & 0xFF
        # if key==ord("q"):
        #     break
        if capture_button:
            captured_image = frame
            break
    cam.release()
    # img = cv2.imdecode(captured_image,1)
    # img = cv2.cvtColor(captured_image,cv2.COLOR_BGR2RGB)
    img = cv2.resize(captured_image, (224,224))
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    index = np.argmax(prediction[0])
    money = money_type[index]
    # st.image(captured_image)
    st.write('This is:', money)
    #cv2.destroyAllWindows()
    
