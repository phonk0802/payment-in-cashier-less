import os
import pickle
from deepface import DeepFace

img_dir = 'Thư mục chứa ảnh khách hàng'
# Ảnh khách có định dạng "tên khách.jpg/png"


def encode_face2(img_path):
    try:
        encode = DeepFace.represent(img_path, enforce_detection=True, model_name='VGG-Face')
    except ValueError:
        print(f'{img_path} cannot detect face!')
        encode = DeepFace.represent(img_path, enforce_detection=False, model_name='VGG-Face')
    return encode[0]['embedding']


def encode_all2(dir_path):
    encoded_list2 = []
    named_list = []
    for img_path in os.listdir(dir_path):
        full_img_path = os.path.join(dir_path, img_path)
        encoded = encode_face2(full_img_path)
        encoded_list2.append(encoded)
        named_list.append(img_path.split('.')[0])
    encoded_list_with_name = [encoded_list2, named_list]
    return encoded_list_with_name

encode_list = []
encode_list2 = []
name_list = []

for path in os.listdir(img_dir):
    full_path = os.path.join(img_dir, path)
    name = path.split('.')[0]
    name_list.append(name)

    encode_list2.append(encode_face2(full_path))

encode_list_with_name2 = [encode_list2, name_list]

file = open('encode_file2.p', 'wb')
pickle.dump(encode_list_with_name2, file)
file.close()
