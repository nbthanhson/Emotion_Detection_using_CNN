import cv2
import os

#Testing the model on some images
categories = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def prepare(img_data):
    img_data = cv2.resize(img_data, (48, 48))
    img_data = img_data.astype('float32')
    img_data /= 255.0
    img_data -= 0.5
    img_data *= 2.0
    return img_data.reshape(-1, 48, 48, 1)

img_text = '/content/gdrive/MyDrive/Emotion Detection/Test/'

count = 1
for f in os.listdir(img_text):
    plt.figure(figsize=(100,100))
    plt.subplot(len(os.listdir(img_text)), 1, count)
    files = os.path.join(img_text, f)
    img = cv2.imread(files)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayImage)
    for (x, y, w, h) in faces:
        img_face = grayImage[y:y+h, x:x+w]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        prediction = model.predict([prepare(img_face)])
        category = categories[np.argmax(prediction)]
        cv2.putText(img, category, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    count += 1
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
