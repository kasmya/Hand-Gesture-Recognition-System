import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('hand_gesture_model.h5')

# Get class labels from the training folder structure
# Replace with your actual class names if needed
class_labels = ['gesture1', 'gesture2', 'gesture3']  # Update accordingly

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    img = cv2.resize(frame, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    preds = model.predict(img)
    class_idx = np.argmax(preds[0])
    class_name = class_labels[class_idx]
    confidence = preds[0][class_idx]
    
    cv2.putText(frame, f'{class_name}: {confidence:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Hand Gesture Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
