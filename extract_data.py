import os
import cv2
import csv
import random
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

def extract_data(file, dest_dir, label):
    dest_file = os.path.join(dest_dir, label + '.csv')
    img = cv2.imread(file)

    if img is None:
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img)

    if results.multi_hand_landmarks is None:
        return

    with open(dest_file, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        for hand_landmarks in results.multi_hand_landmarks:
            data = []
            for landmark in hand_landmarks.landmark:
                data.append(landmark.x)
                data.append(landmark.y)
                data.append(landmark.z)
            csvwriter.writerow(data)


def extract_data_from_files(files, src_dir, dest_dir, label):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for file in files:
        file = os.path.join(src_dir, file)
        extract_data(file, dest_dir, label)

def split_and_extract_data(data_dir, train_dir, test_dir, val_dir, test_percent, val_percent):
    subdirs = [x[0] for x in os.walk(data_dir)]
    subdirs = subdirs[1:]

    for subdir in subdirs:
        files = os.listdir(subdir)
        random.shuffle(files)
        train_index = int(len(files) * (1 - test_percent - val_percent))
        test_index = int(len(files) * test_percent) + train_index
        val_index = int(len(files) * val_percent) + test_index

        train_files = files[:train_index]
        test_files = files[train_index:test_index]
        val_files = files[test_index:val_index]

        label = os.path.basename(subdir)

        extract_data_from_files(train_files, subdir, train_dir, label)
        extract_data_from_files(test_files, subdir, test_dir, label)
        extract_data_from_files(val_files, subdir, val_dir, label)


data_dir = 'sign_language_dataset_A_Z'
train_dir = 'sign/train/'
test_dir = 'sign/test/'
val_dir = 'sign/val/'

split_and_extract_data(data_dir, train_dir, test_dir, val_dir, 0.2, 0.2)
