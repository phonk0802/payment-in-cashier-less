# payment-in-cashier-less
Ứng dụng thị giác máy tính để thanh toán tự động trong cửa hàng không người bán
# Lên hóa đơn tự động bằng Object Detection và Tracking
- Đề tài sử dụng [YOLOv7](https://github.com/WongKinYiu/yolov7) và [SORT](https://github.com/abewley/sort).
- Dữ liệu thực nghiệm là tập 2.301 ảnh bao gồm 9 nhãn tương ứng với các mặt hàng tạp hóa tại Việt Nam.
- Để sử dụng, chạy file retail_detect.py (thay đổi đường dẫn ảnh/video và trọng số mô hình YOLOv7). Kết quả trả về bao gồm 2 file prices.txt và product_log.txt ghi lại những sản phẩm trong ảnh và giá tiền tương ứng.
- Một số kết quả được thể hiện trong thư mục runs/detect/track.

# Thanh toán khuôn mặt bằng Face Recognition
- Đề tài sử dụng OpenCV và VGGFace trong [DeepFace](https://github.com/serengil/deepface) để detect và embedding khuôn mặt.
- Ngoài ra, chúng tôi sử dụng MINIFasNet để thực hiện liveness detection. Chi tiết đọc thêm [tại đây](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing).
- Để sử dụng, trước tiên chạy file encode_gen.py để tạo file encode_file2.p chứa embedding khuôn mặt của toàn bộ khách hàng.
- Sau đó, chạy file customer_recogtion.py để nhận diện khuôn mặt thông qua webcam.
+ Đối với khách hàng cũ, kết quả hiện lên là ảnh khách hàng kèm tên.
+ Đối với khách hàng mới (sau 10s không nhận diện được khách hàng), có thể bấm R để capture frame hiện tại làm ảnh khách hàng.
+ Sau 20s không detect được khuôn mặt nào, webcam sẽ tự động tắt.
- Tên khách hàng và thời điểm thực hiện nhận diện được lưu lại trong customer_log.

## Demo bằng streamlit trong file demo.py
