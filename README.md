Hand Gesture Recognition System
This project implements a real-time hand gesture recognition system using computer vision and
machine learning techniques.
It detects and classifies hand gestures captured via a webcam, aiming to facilitate touchless control
interfaces for smart devices and assistive technologies.
Features:
- Real-time hand gesture detection and classification
- Utilizes OpenCV for image processing
- Employs a custom-trained machine learning model for gesture recognition
- Designed for scalability and integration into various applications
Getting Started:
Prerequisites:
- Python 3.x
- OpenCV
- NumPy
- Any other dependencies listed in requirements.txt
Installation:
1. Clone the repository:
git clone https://github.com/kasmya/Hand-Gesture-Recognition-System.git
2. Navigate to the project directory:
cd Hand-Gesture-Recognition-System

3. Install the required dependencies:
pip install -r requirements.txt
Usage:
1. Train the model (if not already trained):
python train_model.py
2. Run the prediction script:
python prediction.py
The system will activate your webcam. Perform hand gestures in front of the camera to see real-time
predictions.
Project Structure:
- train_model.py: Script to train the gesture recognition model.
- prediction.py: Script to perform real-time gesture recognition using the webcam.
- requirements.txt: List of project dependencies.
Future Work:
- Expand the range of recognizable gestures.
- Integrate with IoT devices for practical applications.
- Enhance the model's accuracy with a larger dataset.
License:
This project is licensed under the MIT License. See the LICENSE file for details.
