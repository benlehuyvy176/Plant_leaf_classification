from numpy.core.fromnumeric import shape
import streamlit as st
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from segment import *

model = tf.keras.models.load_model('model\Model_EfficientNet_20.h5')
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
leaf_type = {'Snake_Plant': 0,
 'acer_negundo': 1,
 'acer_palmatum': 2,
 'aesculus_glabra': 3,
 'aesculus_pavi': 4,
 'asimina_triloba': 5,
 'catalpa_bignonioides': 6,
 'cercis_canadensis': 7,
 'chionanthus_virginicus': 8,
 'eucommia_ulmoides': 9,
 'ferm': 10,
 'gleditsia_triacanthos': 11,
 'lemon': 12,
 'palm': 13,
 'quercus_montana': 14,
 'quercus_muehlenbergii': 15,
 'quercus_stellata': 16,
 'quercus_velutina': 17,
 'tilia_americana': 18,
 'ulmus_glabra': 19}
  

menu = ['camera', 'predict leaf','webcam']
choice = st.sidebar.selectbox('Menu', menu)

def make_cont(img_path, to_gray=False, IMG_SIZE=224):
    """
    Increase image contract
    """
    img = cv2.imread(img_path)
    if to_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    cimg = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), IMG_SIZE / 10), -4, 128)
    return cimg

if choice == 'predict leaf':
    image_upload = st.file_uploader('upload file', type = ['jpg', 'png', 'jpeg'])
    if image_upload != None:
        image_np = np.asarray(bytearray(image_upload.read()),dtype = np.uint8)
        img = cv2.imdecode(image_np,3)
        img = segment_leaf(img,'flood',True,0)
        st.image(img)
        print(img.shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(img.shape)

        img = cv2.resize(img, (224, 224))
        print(img.shape)
        
        img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 224 / 10), -4, 128)
        print(img.shape)        
        img = np.expand_dims(img, axis=0)
        print(img.shape)
        print(img)
        # img = img / 255
        prediction = model.predict(img)
        print(prediction)
        index = np.argmax(prediction[0])
        leaf = list(leaf_type.keys())[index]
        st.image(image_upload)
        st.write('This is:', leaf)
        st.write("Probability: ", np.max(prediction[0],axis=0)*100, "%")
        
        



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
    leaf = leaf_type.keys[index]
    # st.image(captured_image)
    st.write('This is:', leaf)
    
    #cv2.destroyAllWindows()
    
elif choice == 'webcam':
    cap = cv2.VideoCapture(0)
    while(True):
        ret, image_org = cap.read()

        if not ret:
            continue
        # Resize
        # image = image_org.copy()
        # image = cv2.resize(image, (IMG_SIZE,IMG_SIZE),interpolation = cv2.INTER_AREA)
        # Convert to tensor
        # img_array = np.expand_dims(image, axis=0)
        # image_np = np.asarray(bytearray(image_org.read()),dtype = np.uint8)
        image_np = image_org
        # img = cv2.imdecode(image_np,3)
        img = segment_leaf(image_np,'flood',True,0)
        # st.image(img)
        # print(img.shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(img.shape)

        img = cv2.resize(img, (224, 224))
        print(img.shape)
            
        img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 224 / 10), -4, 128)
        print(img.shape)        
        img = np.expand_dims(img, axis=0)
        # Predict
        prediction = model.predict(img)
        a = np.argmax(prediction,axis=1)
        index = np.argmax(prediction[0])
        result = list(leaf_type.keys())[index]
        print("This picture is: ", result)
        print("Probability: ", np.max(prediction[0],axis=0)*100, "%")
        if (np.max(prediction)>=0.8):


            # Show image
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            fontScale = 1.5
            color = (0, 255, 0)
            thickness = 2

            cv2.putText(image_org, result, org, font,
                        fontScale, color, thickness, cv2.LINE_AA)

        cv2.imshow("Picture", image_org)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
