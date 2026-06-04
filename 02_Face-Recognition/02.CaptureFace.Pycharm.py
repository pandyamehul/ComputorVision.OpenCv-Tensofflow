# Importing the necessary libraries
from matplotlib.pyplot import gray

import cv2
import numpy as np
import os

# Load the Haar Cascade face detector
classifier = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")
classifier_eye = cv2.CascadeClassifier("Cascades/haarcascade_eye.xml")

camera = cv2.VideoCapture(0)

sample = 1
n_samples = 25
stop = False
id = input('Type a number (ID): ')

width, height = 220, 220
print("Capturing the faces...")
# prepare output folder: one level above this script, in a `photos` directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
photos_dir = os.path.join(parent_dir, "photos")
os.makedirs(photos_dir, exist_ok=True)

while (True):
    conected, image = camera.read()
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(np.average(image_grey))
    detected_faces = classifier.detectMultiScale(image_grey, scaleFactor=1.5, minSize=(150,150)) 

    for (x, y, l, a) in detected_faces:
       cv2.rectangle(image, (x, y), (x + l, y + a), (0, 0, 255), 2)
       region = image[y:y + a, x:x + l]
       region_eye = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
       detected_eyes = classifier_eye.detectMultiScale(region_eye)
       for (ox, oy, ol, oa) in detected_eyes:
           cv2.rectangle(region, (ox, oy), (ox + ol, oy + oa), (0, 255, 0), 2)

           if cv2.waitKey(1) & 0xFF == ord('c'):
                if np.average(image_grey) > 110:
                    image_face = cv2.resize(image_grey[y:y + a, x:x + l], (width, height))
                    filename = f"person.{id}.{sample}.jpg"
                    filepath = os.path.join(photos_dir, filename)
                    cv2.imwrite(filepath, image_face)
                    print(f"[photo {sample} captured successfully] -> {filepath}")
                    sample += 1
           elif cv2.waitKey(1) & 0xFF == ord('q'):
                stop = True
                break
    
    cv2.imshow("Face", image)
    cv2.waitKey(1)
    if (stop or sample >= n_samples + 1):
        break

print("All faces captured successfully! ")
camera.release()
cv2.destroyAllWindows()