import os
import cv2
import numpy as np
import tensorflow as tf

script_dir = os.path.dirname(os.path.abspath(__file__))

print("script_dir:", script_dir)

# cascade path (relative to workspace root)
cascade_path = os.path.join(script_dir, '..', 'Cascades', 'haarcascade_frontalface_default.xml')

print("cascade_path:", cascade_path)

face_detector = cv2.CascadeClassifier(cascade_path)

# try loading a saved model (JSON + HDF5 weights expected in `Weights` folder)
weights_dir = os.path.join(script_dir, 'Weights')
network_loaded = None
json_path = os.path.join(weights_dir, 'network_emotions.json')
weights_path = os.path.join(weights_dir, 'weights_emotions.hdf5')
try:
  with open(json_path, 'r') as f:
    json_saved_model = f.read()
  network_loaded = tf.keras.models.model_from_json(json_saved_model)
  network_loaded.load_weights(weights_path)
  network_loaded.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
  print('Loaded model from JSON + HDF5')
except Exception as e:
  print('Error loading model:', e)
  network_loaded = None

# emotion labels used in the notebook
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

camera = cv2.VideoCapture(0)

print('Press "q" to quit.')
while True:
  ret, frame = camera.read()
  if not ret:
    break

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

  for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    try:
      roi = cv2.resize(roi_gray, (48, 48))
    except Exception:
      continue

    roi = roi.astype('float32') / 255.0
    roi = np.expand_dims(roi, axis=0)
    roi = np.expand_dims(roi, axis=-1)  # (1,48,48,1)

    label = 'Model not loaded'
    if network_loaded is not None:
      try:
        prediction = network_loaded.predict(roi)
        result = np.argmax(prediction)
        label = emotions[result]
      except Exception as e:
        label = 'Predict error'

    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

  cv2.imshow('Emotion Classification', frame)
  key = cv2.waitKey(1) & 0xFF
  if key == ord('q'):
    break

camera.release()
cv2.destroyAllWindows()