# Ứng dụng Kiểm Tra Văn Bản Trùng Lặp Tiếng Việt

Ứng dụng web này cho phép người dùng kiểm tra độ tương đồng giữa hai đoạn văn bản tiếng Việt và dự đoán xem chúng có trùng lặp hay không, sử dụng ba thuật toán học máy khác nhau: KNN, Naive Bayes và Decision Tree.

## Tính năng

- **Kiểm tra độ tương đồng**: Tính toán bốn loại độ tương đồng khác nhau:
  - Cosine Similarity (sử dụng TF-IDF)
  - Jaccard Similarity
  - Tỷ lệ từ chung
  - Tỷ lệ độ dài

- **Dự đoán trùng lặp**: Sử dụng ba mô hình khác nhau để dự đoán xem hai văn bản có trùng lặp không:
  - K-Nearest Neighbors (KNN)
  - Naive Bayes
  - Decision Tree

- **Xử lý tiếng Việt**: Hỗ trợ xử lý văn bản tiếng Việt với:
  - Chuẩn hóa văn bản (chuyển thành chữ thường, loại bỏ ký tự đặc biệt)
  - Tokenization tiếng Việt (sử dụng thư viện pyvi)

## Cài đặt

### Yêu cầu

- Python 3.6 trở lên
- Các thư viện trong file `requirements.txt`

### Các bước cài đặt

1. Clone repository:
   ```
   git clone https://github.com/your-username/vietnamese-text-duplicate-checker.git
   cd vietnamese-text-duplicate-checker
   ```

2. Tạo và kích hoạt môi trường ảo (tùy chọn nhưng khuyến khích):
   ```
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. Cài đặt các thư viện cần thiết:
   ```
   pip install -r requirements.txt
   ```

## Sử dụng

### Huấn luyện mô hình

Trước khi chạy ứng dụng, bạn cần huấn luyện và lưu các mô hình:

```
python train_models.py
```

Tập lệnh này sẽ:
- Tải dữ liệu từ file `dataset_labeled.csv` (nếu tồn tại)
- Hoặc tạo dữ liệu giả lập nếu file không tồn tại
- Huấn luyện ba mô hình KNN, Naive Bayes và Decision Tree
- Lưu các mô hình vào thư mục `models/`

### Chạy ứng dụng

Sau khi huấn luyện mô hình xong, bạn có thể chạy ứng dụng web:

```
python app.py
```

Sau đó, mở trình duyệt web và truy cập địa chỉ http://127.0.0.1:5000/

### Sử dụng ứng dụng

1. Nhập văn bản thứ nhất vào ô bên trái
2. Nhập văn bản thứ hai vào ô bên phải
3. Nhấn nút "Kiểm tra trùng lặp"
4. Xem kết quả:
   - Các độ tương đồng văn bản
   - Dự đoán từ các mô hình

## Cấu trúc dự án

```
vietnamese-text-duplicate-checker/
├── app.py                    # Ứng dụng Flask chính
├── train_models.py           # Script huấn luyện mô hình
├── requirements.txt          # Các thư viện cần thiết
├── README.md                 # File hướng dẫn
├── models/                   # Thư mục chứa mô hình đã lưu
│   ├── knn_model.pkl
│   ├── nb_model.pkl
│   └── dt_model.pkl
├── static/                   # Tài nguyên tĩnh
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
└── templates/                # Templates HTML
    └── index.html
```

## Tạo file `requirements.txt`

Để đảm bảo ứng dụng hoạt động đúng, bạn cần tạo file `requirements.txt` với nội dung sau:

```
Flask==2.0.1
numpy==1.21.0
pandas==1.3.0
scikit-learn==0.24.2
pyvi==0.1.1
```

## Giấy phép

Dự án này được phân phối dưới giấy phép MIT. Xem file `LICENSE` để biết thêm chi tiết. 