import cv2
import mediapipe as mp

# Initialize MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Start capturing from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip and convert color for OpenCV
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks for visualization
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmarks for each finger tip and base (MCP joint)
            index_finger_tip = hand_landmarks.landmark[
                mp_hands.HandLandmark.INDEX_FINGER_TIP
            ]
            index_finger_mcp = hand_landmarks.landmark[
                mp_hands.HandLandmark.INDEX_FINGER_MCP
            ]
            middle_finger_tip = hand_landmarks.landmark[
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP
            ]
            ring_finger_tip = hand_landmarks.landmark[
                mp_hands.HandLandmark.RING_FINGER_TIP
            ]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            # Check if the index finger is extended
            index_extended = index_finger_tip.y < index_finger_mcp.y

            # Check if other fingers are curled (i.e., their tips are below their base points)
            middle_curled = (
                middle_finger_tip.y
                > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
            )
            ring_curled = (
                ring_finger_tip.y
                > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y
            )
            pinky_curled = (
                pinky_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y
            )

            # Only detect pointing if the index finger is extended and other fingers are curled
            if index_extended and middle_curled and ring_curled and pinky_curled:
                # Calculate position of the index finger tip
                h, w, _ = frame.shape
                index_tip_x, index_tip_y = int(index_finger_tip.x * w), int(
                    index_finger_tip.y * h
                )

                # Indicate pointing gesture detected
                cv2.putText(
                    frame,
                    "Pointing Detected!",
                    (index_tip_x, index_tip_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.circle(frame, (index_tip_x, index_tip_y), 10, (0, 255, 0), -1)

    # Display the frame
    cv2.imshow("Pointing Gesture Detection", frame)

    # Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
