# VIP_Project
In this repository we have uploaded team8 th final project.
import cv2
import numpy as np

# Load face detection model
face_net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

# Load mask detection model
mask_net = cv2.dnn.readNetFromCaffe('mask_deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

# Open the default camera (index 0)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Resize the frame for faster processing
    frame = cv2.resize(frame, (300, 300))

    # Perform face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([300, 300, 300, 300])
            (startX, startY, endX, endY) = box.astype("int")

            # Extract the face region
            face = frame[startY:endY, startX:endX]

            # Perform mask detection
            mask_blob = cv2.dnn.blobFromImage(face, 1.0, (224, 224), (104.0, 177.0, 123.0))
            mask_net.setInput(mask_blob)
            mask_predictions = mask_net.forward()

            # Get the index of the prediction with the highest confidence
            mask_index = np.argmax(mask_predictions[0])

            # Define classes for mask and no mask
            classes = ["Mask", "No Mask"]

            # Display the result
            label = classes[mask_index]
            color = (0, 255, 0) if mask_index == 0 else (0, 0, 255)
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # Display the resulting frame
    cv2.imshow('Mask Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
Explanation:

Loading Models:

We load two pre-trained models for face detection (deploy.prototxt and res10_300x300_ssd_iter_140000.caffemodel) and mask detection (mask_deploy.prototxt and res10_300x300_ssd_iter_140000.caffemodel) using OpenCV's deep neural network (dnn) module.
Camera Initialization:

We open the default camera (index 0) using OpenCV's VideoCapture.
Frame Capture and Resize:

We continuously capture frames from the camera, resize them to 300x300 pixels for faster processing.
Face Detection:

Using the face detection model, we identify faces in the frame. The faces are detected by setting a confidence threshold of 0.5.
Loop Over Detections:

For each detected face, we extract the face region and perform mask detection on that region.
Mask Detection:

The face region is fed into the mask detection model, and predictions are obtained.
Display Results:

Results, including the mask label and bounding box, are displayed on the frame.
User Interface:

The processed frame is displayed in a window named 'Mask Detection'. The loop continues until the 'q' key is pressed.
Cleanup:

The camera is released, and all OpenCV windows are closed.
Reasons for Choosing Face Detection:

Public Health Relevance:

Face detection is a crucial step for enforcing mask-wearing in public spaces, contributing to public health and safety.
Efficient Mask Region Processing:

By detecting faces first, we can focus mask detection efforts on the regions where masks are expected, improving computational efficiency.
Real-time Application:

Face detection allows for real-time processing, making the system suitable for applications like monitoring public spaces and workplaces.
