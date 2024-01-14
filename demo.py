import os
import shutil

import cv2
import streamlit as st
import pandas as pd
from io import StringIO
from retail_detect import get_shopping_list_from_source
from customer_recognition import predict_face


def rename_cus():
    name = st.text_input("Nhập tên của bạn:")
    st.write("Giá trị nhập vào là:", name)
    path = 'image/img2'
    if st.button("đã xong"):
        new_path = path + '/' + str(name) + '.jpg'
        shutil.move('image/temp_frame.jpg', new_path)


def main():
    st.title("Welcome")
    uploaded_file = st.file_uploader("Chọn video hoặc ảnh", type=["mp4", "png", "jpg", "jpeg"])

    if uploaded_file:
        file_path = uploaded_file.name
        full_path = os.path.join('real_test', file_path)
        get_shopping_list_from_source(full_path)
        if '.jpg' in file_path:
            st.image('runs/detect/track/' + file_path, caption='Kết quả', use_column_width=True)
        file_path = 'prices.txt'

        with open(file_path, "r", encoding='utf8') as file:
            text_content = file.read()
        lines = text_content.splitlines()
        sum = 0
        for i in lines:
            a = i.split(',')
            sum += int(a[2]) * int(a[1])

        df = pd.read_csv(StringIO(text_content), names=["Tên sản phẩm", "Số lượng", "Giá tiền/1 sản phẩm", "Tổng giá"], sep=',')
        st.title("Hóa đơn của bạn:")
        st.markdown(
            f'<div style="text-align:center; width: 1000px">{df.to_html(index=False)}</div>',
            unsafe_allow_html=True
        )
        st.write(f'Tổng tiền: {sum} đồng')
        col1, col2 = st.columns([1, 2])

        # Nút "Chuyển đến thanh toán"
        if 'face' not in st.session_state:
            st.session_state.face = None

        if col1.checkbox("Thanh toán"):
            try:
                if st.session_state.face is None:
                    st.session_state.face = predict_face()
                if st.session_state.face == -1:
                    rename_cus()
            except cv2.error:
                st.write('Đã kết thúc việc thanh toán!')

        # Nút "Quay lại"
        if col2.checkbox("Chụp lại"):
            st.warning("Xác nhận. Bạn có thể chụp lại!")


if __name__ == "__main__":
    main()
