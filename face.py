import cv2  # Import OpenCv library
from fer import FER  # OpenCV Facial Expression Recognition pre trained model


# Load the Haar Cascade pre-trained model from OpenCV for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Initialize emotion detector
emotion_detector = FER()

# Open a connection to the webcam (0 = default webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Captures the frames

    # Convert the frame to grayscale because it is simpler and eaiser to be analyzed by facial recognition
    # functions that don't need color as an attribute
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame, and stores their position (topLeftXPos, topLeftYPos, width, height) in a list
    # scaleFactor determines the % that the image size is reduced at each step of the function to determine a face
    # minNeighbors determines the min amount of overlap needed for an area to be considered a face
    # minSize is the smallest size a face can be (in pixels)
    faces = face_cascade.detectMultiScale(
        grayscale, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30)
    )

    # Loops through the lists of face coordinates, and draws a rectangle around the face
    # Arguments: (the frame, top left corner, bottom right corner, color (BGR format), thickness of line)
    for x, y, w, h in faces:
        # Crop face from the frame
        face_frame = frame[y : y + h, x : x + w]

        # Detect emotion using the FER model
        emotion, score = emotion_detector.top_emotion(face_frame)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Check if score is None before formatting it
        if score is not None:
            cv2.putText(
                frame,
                f"{emotion}: {score:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )
        else:
            cv2.putText(
                frame,
                f"{emotion}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )

    # Display the resulting frame
    cv2.imshow("Face & Mood Detection Proptotype Program #2", frame)

    # Waits 1ms between frames to see if user presses 'q' on keyboard
    if cv2.waitKey(1) == ord("q"):
        break  # Exits the loop, ending the code

# Cut the connection between the webcam and close all windows opened during the program
cap.release()
cv2.destroyAllWindows()
