import cv2
import numpy as np


class HandGestureRecognizer:

    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        self.gestures = ["fist", "palm", "index", "ok"]
        self.model = cv2.CascadeClassifier("haarcascade_hand.xml")

    def track_hand(self):
        # Capture the current frame from the camera.
        ret, frame = self.camera.read()

        # Convert the frame to grayscale.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect the hands in the frame.
        hands = self.model.detectMultiScale(gray, 1.3, 5)

        # If hands are detected, draw them on the frame.
        for hand in hands:
            x, y, w, h = hand
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Return the frame with the hands drawn on it.
        return frame

    def recognize_gesture(self, frame):
        # Convert the frame to grayscale.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the contours of the hands in the frame.
        contours, hierarchy = cv2.findContours(
            gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour, which is the contour of the hand.
        largest_contour = max(contours, key=cv2.contourArea)

        # Find the center of the hand.
        center = np.array(largest_contour).mean(axis=0)

        # Calculate the distance between the center of the hand and the top of the hand.
        distance = np.linalg.norm(center - np.array([0, 0]))

        # Classify the gesture based on the distance.
        if distance < 100:
            gesture = "fist"
        elif distance < 200:
            gesture = "palm"
        elif distance < 300:
            gesture = "index"
        else:
            gesture = "ok"

        return gesture


def main():
    recognizer = HandGestureRecognizer()

    while True:
        # Get the current frame from the camera.
        frame = recognizer.track_hand()

        # Recognize the gesture in the frame.
        gesture = recognizer.recognize_gesture(frame)

        # Display the gesture on the frame.
        cv2.putText(frame, gesture, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

        # Display the frame.
        cv2.imshow("Hand Gesture Recognition", frame)

        # Wait for a key press.
        key = cv2.waitKey(1)

        # If the key is ESC, break out of the loop.
        if key == 27:
            break

    # Close the camera.
    recognizer.camera.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
