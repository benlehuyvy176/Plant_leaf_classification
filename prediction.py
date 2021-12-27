import numpy as np
import cv2
import tensorflow as tf

IMG_SIZE = 224
cap = cv2.VideoCapture(0)

#Load your model and check create the  list
Model_Path = 'Model_1.h5'
class_names = ["1,000","10,000","100,000","2,000","20,000","200,000","5,000","50,000","500,000"]
my_model = tf.keras.models.load_model(Model_Path)

while(True):
    # Capture frame-by-frame
    ret, image_org = cap.read()

    if not ret:
        continue
    # Resize
    image = image_org.copy()
    image = cv2.resize(image, (IMG_SIZE,IMG_SIZE),interpolation = cv2.INTER_AREA)
    # Convert to tensor
    img_array = np.expand_dims(image, axis=0)
     # Predict
    prediction = my_model.predict(img_array)
    a = np.argmax(prediction,axis=1)
    result = class_names[int(a)]
    print("This picture is: ", result)
    print("Probability: ", np.max(prediction[0],axis=0)*100, "%")
    # if (np.max(prediction)>=0.5) and (np.argmax(prediction[0]!=0)):
    if (np.max(prediction)>=0.7):


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

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()