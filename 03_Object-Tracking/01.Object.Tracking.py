import cv2
import os
import sys

#tracker = cv2.TrackerKCF_create()
tracker = cv2.TrackerCSRT_create()

video_path = 'videos/street.mp4'
video = cv2.VideoCapture(video_path)
# use below for camera
# video = cv2.VideoCapture(0)

if not video.isOpened():
    print(f"Could not open '{video_path}'. Trying camera 0.")
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print('ERROR: Could not open video file or camera.')
        sys.exit(1)

ok, frame = video.read()

if not ok or frame is None or getattr(frame, 'size', 0) == 0:
    print('ERROR: Failed to read first frame from source.')
    sys.exit(1)

# Let user select ROI on the first frame. Provide window name so imshow works.
bbox = cv2.selectROI('Select ROI', frame, showCrosshair=True, fromCenter=False)

if bbox == (0, 0, 0, 0):
    print('No ROI selected. Exiting.')
    sys.exit(0)

ok = tracker.init(frame, bbox)

while True:
    ok, frame = video.read()
    if not ok or frame is None:
        break
    
    ok, bbox = tracker.update(frame)

    if ok:
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2, 1)
    else:
        cv2.putText(frame, 'Error', (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == 27: # ESC
        break

video.release()
cv2.destroyAllWindows()
