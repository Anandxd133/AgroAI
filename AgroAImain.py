import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

model = load_model('leaf_deficiency_model.h5')

# Defining a dictionary to map the model's output to nutrient deficiencies
deficiency_map = {
    0: 'Healthy',
    1: 'Nitrogen Deficiency',
    2: 'Phosphorous Deficiency',
    3: 'Potassium Deficiency'
}

# image preprocessing
def preprocess_image(image):
    image = cv2.resize(image, (128, 128)) 
    image = img_to_array(image) / 255.0  
    image = np.expand_dims(image, axis=0)  
    return image

# nutrient deficiency detection
def detect_deficiency(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    deficiency = deficiency_map[predicted_class]
    return deficiency


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    deficiency = detect_deficiency(frame)
    cv2.putText(frame, f'Deficiency: {deficiency}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Leaf Nutrient Deficiency Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
