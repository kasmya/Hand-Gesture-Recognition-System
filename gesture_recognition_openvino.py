import cv2
import numpy as np
from openvino.runtime import Core

core = Core()
model = core.read_model(model="optimized_model/hand_gesture_model.xml")
compiled_model = core.compile_model(model, device_name="CPU")

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

class_labels = ['gesture1', 'gesture2', 'gesture3']

_, _, h, w = input_layer.shape

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0).astype(np.float32)

    result = compiled_model([img])[output_layer]
    class_idx = np.argmax(result[0])
    confidence = result[0][class_idx]
    class_name = class_labels[class_idx]

    cv2.putText(frame, f'{class_name}: {confidence:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
