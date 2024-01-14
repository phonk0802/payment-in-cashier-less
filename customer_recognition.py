from datetime import datetime
import csv
import os
import pickle
import time
import cv2
from matplotlib import pyplot as plt
import numpy as np
from deepface import DeepFace
from Silent_Face_Anti_Spoofing.liveness_predict import test_face

file = open('encode_file2.p', 'rb')
encode_list_known = pickle.load(file)
file.close()
print('Encode Loaded!')
encode_list, name = encode_list_known
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

start_time = time.time()
display_time = 2
fc = 0
FPS = 0
frame_count = 0
img_cus_list = []
unknown_start_time1 = None
unknown_start_time2 = None
new_cus_add = False

for path in os.listdir(image_dir):
    full_path = os.path.join(image_dir, path)
    img_temp = cv2.imread(full_path)
    img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2RGB)
    img_cus_list.append(img_temp)


def get_label(img):
    x = test_face(img,
             model_dir='Silent_Face_Anti_Spoofing/resources/anti_spoof_models',
             device_id=0)
    label, value, speed, box = x
    print(box)
    print(label)
    return label


def compare_all(encoded_frame, encoded_list, threshold=0.2):
    res = 'Unknown'
    encoded_frame = np.array(encoded_frame)

    for i in range(len(encoded_list)):
        encoded = np.array(encoded_list[i])
        distance = np.linalg.norm(encoded - encoded_frame)
        if distance < threshold:
            res = i
    return res


def make_result_img(ide):
    img_cus = img_cus_list[ide]
    if img_cus.shape[0] < 320:
        img_cus = cv2.resize(img_cus, None, fx=2, fy=2)
    elif img_cus.shape[0] > 720:
        img_cus = cv2.resize(img_cus, None, fx=0.5, fy=0.5)
    img_cus = cv2.cvtColor(img_cus, cv2.COLOR_BGR2RGB)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 1
    font_color = (255, 0, 0)
    font_thickness = 2
    text_size = cv2.getTextSize("Customer's name: " + name[ide], font, font_size, font_thickness)[0]
    text_position_x = (img_cus.shape[1] - text_size[0]) // 2
    text_position_y = img_cus.shape[0] - 20
    rect_color = (200, 200, 200)  # Màu của hình chữ nhật
    rect_coords = [(text_position_x - 5, text_position_y - text_size[1] - 5),
                   (text_position_x + text_size[0] + 5, text_position_y + 5)]
    cv2.rectangle(img_cus, rect_coords[0], rect_coords[1], rect_color, -1)  # -1 để vẽ hình chữ nhật đầy đủ
    cv2.putText(img_cus, "Customer's name: " + name[ide], (text_position_x, text_position_y), font, font_size,
                font_color, font_thickness)
    cv2.imshow('Customer Image', img_cus)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def predict_face():
    cus_name = 'Cannot detect any face!'
    COL_NAMES = ['NAME', 'TIME']
    start_time = time.time()
    display_time = 2
    fc = 0
    FPS = 0
    frame_count = 0
    unknown_start_time1 = None
    unknown_start_time2 = None
    new_cus_add = False
    fake_person = False
    counter = 0
    while True:
        success, img = cap.read()
        fc += 1
        frame_count += 1
        TIME = time.time() - start_time
        if TIME >= display_time:
            FPS = fc / TIME
            fc = 0
            start_time = time.time()

        fps_disp = "FPS: " + str(FPS)[:5]
        # img_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        img_small = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imsave('temp_frame.jpg', img_small)
        if success:
            if counter % 30 == 0:
                try:
                    check_img = cv2.imread(
                        'temp_frame.jpg')
                    label = get_label(check_img)
                    # print(label)
                    if label != 1:
                        fake_person = True
                    else:
                        fake_person = False
                        encode_per_frame = DeepFace.represent(
                            "temp_frame.jpg",
                            enforce_detection=True)
                        result = compare_all(encode_per_frame[0]['embedding'], encode_list, 0.39)
                        if result != 'Unknown':
                            cus_name = name[int(result)]
                            date = datetime.fromtimestamp(time.time()).strftime("%d-%m-%y")
                            timestamp = datetime.fromtimestamp(time.time()).strftime("%H:%M-%S")
                            exist = os.path.isfile('customer_log\\log' + date + '.csv')
                            if os.path.exists('product_log.txt'):
                                with open('product_log.txt', 'r', encoding='utf8') as file:
                                    product, price = file.read().split(';')
                                customer_log = [str(cus_name), str(timestamp), product, price]
                            else:
                                COL_NAMES = ['NAME', 'TIME']
                                customer_log = [str(cus_name), str(timestamp)]
                            if exist:
                                with open('customer_log\\log' + date + '.csv', '+a') as csvfile:
                                    writer = csv.writer(csvfile)
                                    writer.writerow(customer_log)
                                csvfile.close()
                            else:
                                with open('customer_log\\log' + date + '.csv', '+a') as csvfile:
                                    writer = csv.writer(csvfile)
                                    writer.writerow(COL_NAMES)
                                    writer.writerow(customer_log)
                                csvfile.close()
                            time.sleep(1)
                            make_result_img(int(result))
                            cap.release()
                            break

                        else:
                            cus_name = result
                            if unknown_start_time1 is None:
                                unknown_start_time1 = time.time()
                            elif time.time() - unknown_start_time1 > 5 and not fake_person:
                                new_cus_add = True

                except ValueError:
                    if not unknown_start_time2:
                        unknown_start_time2 = time.time()
                    elif time.time() - unknown_start_time2 > 20:
                        print("Cannot detect any face for more than 20 seconds. Turning off the webcam...")
                        cap.release()
                        break
                    pass
                except cv2.error:
                    print('Run this file twice in a row')
            counter += 1
            if fake_person:
                cv2.putText(img, 'Maybe you are fake or it is too dark', (100, 70),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 3)
            else:
                img = cv2.putText(img, cus_name, (10, img.shape[0] - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                img = cv2.putText(img, fps_disp, (10, 25),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if new_cus_add:
                k = cv2.waitKey(1)
                if k != ord('q'):
                    cv2.putText(img, 'Press Q if you wanna quit :<', (100, 70),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 3)
                else:
                    cap.release()
                    break
                if k != ord('r'):
                    cv2.putText(img, 'Press R to capture your face', (100, 100),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 3)

                else:
                    print('New customer added!')
                    cap.release()
                    return -1

        cv2.imshow('Face', img)
        k = cv2.waitKey(1)
        if k == ord('q'):
            cap.release()
            break
predict_face()
