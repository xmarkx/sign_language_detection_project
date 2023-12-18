import os
import cv2
import keyboard
import numpy as np
from tensorflow.keras.utils import to_categorical
from detector import mp_holistic, mediapipe_detection, draw_landmarks, extract_keypoints


class Collector:
    def __init__(self):
        self.actions = ['a', 'b', 'c'] # ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        self.no_sequences = 30
        self.sequence_length = 30
        self.data_path = os.getcwd() + '\\MP_Data'

    def folder_setup(self, actions_in=False, no_sequences_in=False):
            """
            actions = np.array(['hello', 'thanks', 'iloveyou'])
            no_sequences = 30
            sequence_length = 30
            """
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
            # Actions that we try to detect
            if actions_in:
                assert actions_in, "Please give a list of classes in string format"
                self.actions = np.array(actions_in)
            # Thirty videos worth of data
            if no_sequences_in:
                assert no_sequences_in > 0, "Please define the number (int format) of sequences / videos to record per class"
                self.no_sequences = no_sequences_in

            for action in self.actions:
                for sequence in range(1,self.no_sequences+1):
                    try: 
                        os.makedirs(os.path.join(self.data_path, action, str(sequence)))
                    except:
                        pass
    
    def collect_data(self):
        cap=cv2.VideoCapture(1)
        if not cap.isOpened():
            cap=cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Cannot Open WebCam")
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            print(f'collect_data actions: {self.actions}')
            # Loop through actions
            for action in self.actions:
                print(f'for action in self.actions: {action}')
                # Loop through sequences aka videos
                for sequence in range(self.no_sequences):
                    print(f'total sequences: {self.no_sequences}')
                    print(f'for sequence in range(self.no_sequences): {sequence}')
                    # Wait for 'n' key press to start collecting frames
                    print("Press 'n' to start collecting frames for {} Video Number {}".format(action, sequence))
                    keyboard.wait('n')
                    # Loop through video length aka sequence length
                    for frame_num in range(self.sequence_length):
                        print(f'for frame_num in range(self.sequence_length): {frame_num}')
                        if not cap.isOpened():
                            cap=cv2.VideoCapture(0)
                        if not cap.isOpened():
                            raise IOError("Cannot Open WebCam")
                        # Read feed
                        ret, frame = cap.read()
                        # print(frame)
                        # print(action)
                        # Make detections
                        image, results = mediapipe_detection(frame, holistic)

                        # Draw landmarks
                        draw_landmarks(image, results, face=False, pose=False)

                        # NEW Apply wait logic
                        if frame_num == 0: 
                            cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                            cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            # Show to screen
                            cv2.imshow('OpenCV Feed', image)
                            cv2.waitKey(1000)
                        else: 
                            cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence+1), (15,12), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            # Show to screen
                            cv2.imshow('OpenCV Feed', image)

                        # NEW Export keypoints
                        keypoints = extract_keypoints(results)
                        npy_path = os.path.join(self.data_path, action, str(sequence+1), str(frame_num+1) + '.npy')
                        # print(npy_path)
                        np.save(npy_path, keypoints)

                        # Break gracefully
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break

            cap.release()
            cv2.destroyAllWindows()

    def preprocess(self):
        label_map = {label:num for num, label in enumerate(self.actions)}
        print(label_map)
        sequences, labels = [], []
        for action in self.actions:
            for sequence in np.array(os.listdir(os.path.join(self.data_path, action))).astype(int):
                window = []
                for frame_num in range(self.sequence_length):
                    res = np.load(os.path.join(self.data_path, action, str(sequence), "{}.npy".format(frame_num+1)))
                    res = res.flatten()
                    window.append(res)
                sequences.append(window)
                labels.append(label_map[action])
        # print(np.array(sequences).shape)
        # print(to_categorical(labels).astype(int).shape)
        return np.array(sequences), to_categorical(labels).astype(int)