import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def count_and_label_fingers(landmarks, frame):
    labels = []
    positions = []

    # Thumb (labeled as 5)
    if landmarks[4].x < landmarks[3].x:
        labels.append(5)
        positions.append((landmarks[4].x, landmarks[4].y))

    # Index, middle, ring, and pinky fingers
    for i, tip in enumerate([8, 12, 16, 20], start=1):  # Start counting from 1 for the index finger
        if landmarks[tip].y < landmarks[tip - 2].y:
            labels.append(i)  # Label the finger with the appropriate number
            positions.append((landmarks[tip].x, landmarks[tip].y))

    # Label each raised finger
    h, w, _ = frame.shape
    for i, (x, y) in enumerate(positions):
        px, py = int(x * w), int(y * h)
        cv2.putText(frame, str(labels[i]), (px, py - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return len(labels)  # Return the total count of raised fingers

def main():
    # Access the webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect hands
        result = hands.process(rgb_frame)

        num_fingers = 0  # Initialize finger count
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Count and label the raised fingers
                num_fingers = count_and_label_fingers(hand_landmarks.landmark, frame)

        # Display the total finger count in the top-left corner
        cv2.putText(frame, f'Total Fingers: {num_fingers}', (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow('Hand Keypoint Detection', frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
