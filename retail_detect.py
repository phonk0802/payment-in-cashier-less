import glob
import os
from pathlib import Path
import cv2
from matplotlib import pyplot as plt

from yolov7.models.experimental import attempt_load
from yolov7.utils.torch_utils import select_device
from yolov7.detect_and_track import detect_id

weights = 'test_886.pt'
#device = select_device('cpu')
#model = attempt_load(weights, map_location=device)  # load FP32 model

nc = 9
price_dict = dict(poca=10000, custas=12000, milo=8000, omachi=5000, fami=6000, cafe=10000, pen=6000, hao_hao=3000,
                  bottle=60000)

name_dict = dict(poca='Bim bim Poca/Lays', custas='Hộp bánh Custas (2 cái)', milo='Hộp sữa Milo', omachi='Mì Omachi',
                 fami='Hộp sữa Fami', cafe='Cà phê lon Boss', pen='Bút bi FlexOffice', hao_hao='Mì Hảo Hảo',
                 bottle='Bình nuớc')


def check_index(index, dictionary):
    for key, value in dictionary.items():
        if index in value:
            value.remove(index)


def get_shopping_list_from_txt(file_path, type):
    shopping_dict = {}

    with open(file_path, 'r') as total_file:
        lines = total_file.readlines()
        for line in lines:
            if len(line) > 2:
                index_str, label_str, x_centre = line.split()[0:3]
                conf = float(line.split()[-1])
                max_conf = 0

                if 0.25 < float(x_centre) < 0.4 or type != 'mp4':
                    index = int(index_str)
                    label = int(label_str)
                    check_index(index, shopping_dict)
                    # else:
                    max_conf = conf
                    if label in shopping_dict:
                        shopping_dict[label].append(index)
                    else:
                        shopping_dict[label] = [index]
    for label in shopping_dict.keys():
        shopping_dict[label] = list(set(shopping_dict[label]))

    for i in range(nc):
        if i not in shopping_dict.keys():
            shopping_dict[i] = []
    return shopping_dict


def get_shopping_list_from_source(source):
    file_name = Path(source).stem
    if os.path.exists(f'runs\\detect\\track\\labels\\{file_name}'):
        for filename in os.listdir(f'runs\\detect\\track\\labels\\{file_name}'):
            if filename.endswith(".txt"):
                file_path = os.path.join(f'runs\\detect\\track\\labels\\{file_name}', filename)
                os.remove(file_path)
    detect_id(source=source)
    with open(f'{file_name}.txt', 'w') as total_file:
        files = glob.glob(f'runs\\detect\\track\\labels\\{file_name}\\*.txt')

        for f in files:
            with open(f, 'r') as file:
                content = file.read()
                total_file.write(content + '\n')

    path_txt = f'{file_name}.txt'
    result = get_shopping_list_from_txt(path_txt, type=source.split('.')[-1])
    mapping = {0: 'poca', 1: 'custas', 2: 'milo', 3: 'omachi', 4: 'fami', 5: 'cafe', 6: 'pen', 7: 'hao_hao',
               8: 'bottle'}
    class_result = {mapping[key]: len(value) for key, value in result.items()}
    # print(result)
    # print("Kết quả dự đoán", str(set(class_result.items())))
    # with open(f'real_test_2_side_2\\{file_name}.txt', 'r') as file_label:
    #    label = file_label.read()
    # label = eval(label)
    # print(type(label))
    # print("Kết quả thực tế", label)
    # print(set(class_result.items()) == label)

    print("Danh sách hàng hóa đã mua là:")
    product_list = []
    total_price_all_products = 0

    for product, quantity in class_result.items():
        if product in price_dict:
            total_price = quantity * price_dict[product]
            if quantity > 0:
                print(f'{name_dict[product]}, số lượng: {quantity}, giá: {total_price}, tổng {total_price*quantity}')
            product_list.append({'product': product, 'total_price': total_price})
            total_price_all_products += total_price
    print(f'Tổng hóa đơn là: {total_price_all_products}')
    with open('prices.txt', 'w', encoding='utf8') as file:
        for product in class_result:
            if class_result[product] > 0:
                file.write(f'{name_dict[product]},{class_result[product]},{price_dict[product]},{price_dict[product]*class_result[product]}\n')
    with open('product_log.txt', 'w', encoding='utf8') as file:
        file.write(f'{class_result};{total_price_all_products}')
    return class_result, total_price_all_products
product_path = "đường dẫn đến ảnh.jpg hoặc video.mp4"
get_shopping_list_from_source(product_path)
