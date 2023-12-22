import cv2
import numpy as np
import random
from tensorflow.keras.models import load_model
from collector import Collector 
from detector import mp_holistic, mediapipe_detection, draw_landmarks, extract_keypoints


def prob_viz(res, actions, input_frame, colors):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            cv2.rectangle(output_frame, (0,60+num*15), (int(prob*100), 45+num*15), colors[num], -1)
            cv2.putText(output_frame, actions[num], (0, 55+num*15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)

        return output_frame

def run_inference(model_name, num_of_classes, collector=Collector()):
    # Inference config
    model = load_model(model_name, compile=True)
    model.summary()
    # collector = Collector()

    ###### make a list comprehension where we get len(actions) or y_train.shape[-1] number of tuples, with random 3 RGB values between 0-255
    colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for i in range(num_of_classes)]

    # 1. New detection variables
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.75

    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
        while cap.isOpened():
            # Read feed
            ret, frame = cap.read()
            # Make detections
            image, results = mediapipe_detection(frame, holistic)

            # Draw landmarks
            draw_landmarks(image, results, styled=True, face=False, pose=False, left_hand=True, right_hand=True)

            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))


            #3. Viz logic
                if np.unique(predictions[-10:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 

                        if len(sentence) > 0: 
                            if collector.actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(collector.actions[np.argmax(res)])
                        else:
                            sentence.append(collector.actions[np.argmax(res)])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                # Viz probabilities
                image = prob_viz(res, collector.actions, image, colors)

            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()