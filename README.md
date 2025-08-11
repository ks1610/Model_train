Đây là mã nguồn mô hình dự đoán độ ngọt quả cam

File:
1. model.py để tạo file model có thể sử dụng sau này
2. load_model.py để load mô hình và có thể sử dụng để dự đoán ảnh. 
Tuy nhiên sẽ chỉ nhận 1 ảnh từ 1 camera còn việc đọc từ nhiều camera sẽ phải
đa luồng song song nhiều camera và đưa ra kết quả cuối từ các dự đoán của các mô hình.
3. data.pickle là file dữ liệu ảnh