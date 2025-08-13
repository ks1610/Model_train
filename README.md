Đây là mã nguồn mô hình dự đoán dựa trên hình ảnh được cung cấp cho huấn luyện

File:
1. create_pickle.py tạo file .pickle từ các hình ảnh để lưu dữ liệu 
2. data.pickle là file dữ liệu ảnh
3. model.py để tạo file model có thể sử dụng sau này
4. load_model.py để load mô hình và có thể sử dụng để dự đoán ảnh. 
Tuy nhiên sẽ chỉ nhận 1 ảnh từ 1 camera còn việc đọc từ nhiều camera sẽ phải
đa luồng song song nhiều camera và đưa ra kết quả cuối từ các dự đoán của các mô hình.
5. model.pth lưu lại kết quả trainning chạy từ file model.py
6. Camera.py để nhận diện vật thể thông qua camera 

Folder:
1. data lưu các hình ảnh để trainning
2. test các hình ảnh dùng để kiểm tra model với file load_model.py

Khởi động model:
1. Mục đích sử dụng:
- Chạy file **load_model.py** hoặc **Camera.py** để chẩn đoán với hình ảnh hoặc camera (thay đường dẫn hình ảnh dự đoán với file **load_model.py**)

2. Mục đích huấn luyện:
Hiện tại model được tạo để nhận diện quả chuối và quả cam nên để huấn luyện theo mục đích cá nhân cần:

- Thêm các ảnh vào **folder data** theo cấu trúc sau:
    data
    --folder1
    ----ảnh_1.png
    ----ảnh_2.png
    ----ảnh_3.png
    ----ảnh_4.png
    --folder2
    ----ảnh_1.png
    ----ảnh_2.png
    ----ảnh_3.png
    ----ảnh_4.png
lưu ý: mỗi folder ít nhất là 4 ảnh 
- Chạy file **create_pickle.py** để lưu data ảnh vào file data.pickle 
- Chạy file model.py để bắt đầu huấn luyện (Quá trình huấn luyện sẽ phụ thuộc vào lượng data của file data.pickle vừa tạo)
- Chạy file **load_model.py** hoặc **Camera.py** để chẩn đoán với hình ảnh hoặc camera (thay đường dẫn hình ảnh dự đoán với file **load_model.py**)