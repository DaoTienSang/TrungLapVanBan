import pandas as pd
import numpy as np
import re
import os
import pickle
from pyvi import ViTokenizer
from underthesea import text_normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix

print("Bắt đầu huấn luyện các mô hình...")

# 1. Tạo thư mục lưu mô hình nếu chưa tồn tại
MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    print(f"Đã tạo thư mục {MODEL_DIR}")

# 2. Danh sách stopwords và từ điển viết tắt
danhSachStopwords = [
    'và', 'của', 'có', 'là', 'được', 'trong', 'đã', 'với', 'để', 'không', 'cho', 'này', 'khi',
    'tại', 'đến', 'từ', 'theo', 'người', 'những', 'như', 'về', 'nhưng', 'một', 'các', 'vào', 
    'bị', 'còn', 'phải', 'nên', 'lại', 'đang', 'thì', 'rằng', 'mà', 'đó', 'cũng', 'sẽ', 'vì',
    'ra', 'nếu', 'làm', 'thể', 'bởi', 'bằng', 'trên', 'dưới', 'rồi', 'mới', 'lên', 'xuống'
]

tuDienVietTat = {
    'sv': 'sinh viên', 'gv': 'giảng viên', 'đh': 'đại học', 'tphcm': 'thành phố hồ chí minh',
    'vn': 'việt nam', 'ko': 'không', 'k': 'không', 'kq': 'kết quả', 'bt': 'bài tập',
    'ng': 'người', 'tgian': 'thời gian', 'hnay': 'hôm nay', 'tks': 'cảm ơn', 'lm': 'làm'
}

# 3. Hàm tiền xử lý văn bản
def chuanHoaDau(vanBan):
    try:
        return text_normalize(vanBan)
    except:
        return vanBan

def moRongVietTat(vanBan):
    cacTu = vanBan.split()
    for i, tu in enumerate(cacTu):
        tuVietThuong = tu.lower()
        if tuVietThuong in tuDienVietTat:
            cacTu[i] = tuDienVietTat[tuVietThuong]
    return ' '.join(cacTu)

def xuLyNangCao(vanBan):
    vanBan = str(vanBan).lower()
    vanBan = chuanHoaDau(vanBan)
    vanBan = moRongVietTat(vanBan)
    vanBan = re.sub(r"[^a-z0-9À-ỹ\s]", " ", vanBan)
    vanBan = re.sub(r"\s+", " ", vanBan).strip()
    
    return vanBan

def tachTuVaLoaiStopwords(vanBan):
    daTachTu = ViTokenizer.tokenize(vanBan)
    cacTu = daTachTu.split()
    cacTuLoaiStopwords = [tu for tu in cacTu if tu.lower() not in danhSachStopwords]
    
    return ' '.join(cacTuLoaiStopwords)

# 4. Tính độ tương đồng giữa hai văn bản
def tinh_do_tuong_dong(a, b):
    # Cosine similarity với TF-IDF
    tf_idf = TfidfVectorizer(ngram_range=(1, 2))
    tf_idf_matrix = tf_idf.fit_transform([a, b])
    cosine = cosine_similarity(tf_idf_matrix[0:1], tf_idf_matrix[1:2])[0][0]
    
    # Jaccard similarity
    set_a = set(a.split())
    set_b = set(b.split())
    
    union = set_a.union(set_b)
    if len(union) > 0:
        jaccard = len(set_a.intersection(set_b)) / len(union)
    else:
        jaccard = 0
    
    # Tỷ lệ từ chung
    min_len_set = min(len(set_a), len(set_b))
    if min_len_set > 0:
        common_ratio = len(set_a.intersection(set_b)) / min_len_set
    else:
        common_ratio = 0
    
    # Tỷ lệ độ dài
    max_len_text = max(len(a), len(b))
    if max_len_text > 0:
        len_ratio = min(len(a), len(b)) / max_len_text
    else:
        len_ratio = 0
    
    return [cosine, jaccard, common_ratio, len_ratio]

# 5. Trích xuất đặc trưng
def trich_xuat_dac_trung(df):
    features = []
    for _, row in df.iterrows():
        text_a = row['Text_Token_NoStop_A']
        text_b = row['Text_Token_NoStop_B']
        
        # Tính toán các đặc trưng tương đồng
        features.append(tinh_do_tuong_dong(text_a, text_b))
    
    return np.array(features)

# 6. Tải dữ liệu từ file hoặc tạo dữ liệu demo
try:
    print("Đang tải dữ liệu đã gán nhãn...")
    df = pd.read_csv("dataset_labeled.csv")
    print(f"Đã tải xong dữ liệu với {len(df)} mẫu.")
except FileNotFoundError:
    try:
        print("Không tìm thấy file dữ liệu đã gán nhãn. Tải dữ liệu từ data_vietnam.csv...")
        df = pd.read_csv("data_vietnam.csv")
        print(f"Đã tải xong dữ liệu từ data_vietnam.csv với {len(df)} mẫu.")
        
        print("Đang xử lý văn bản tiếng Việt...")
        
        # Tiền xử lý văn bản
        df['Text_Clean_A'] = df['Text_A'].apply(xuLyNangCao)
        df['Text_Clean_B'] = df['Text_B'].apply(xuLyNangCao)
        
        # Tokenization và loại bỏ stopwords
        df['Text_Token_A'] = df['Text_Clean_A'].apply(ViTokenizer.tokenize)
        df['Text_Token_B'] = df['Text_Clean_B'].apply(ViTokenizer.tokenize)
        
        df['Text_Token_NoStop_A'] = df['Text_Clean_A'].apply(tachTuVaLoaiStopwords)
        df['Text_Token_NoStop_B'] = df['Text_Clean_B'].apply(tachTuVaLoaiStopwords)
        
        # Tính độ tương đồng
        print("Đang tính độ tương đồng giữa các cặp văn bản...")
        X = trich_xuat_dac_trung(df)
        
        # Gán nhãn bằng KMeans
        print("Đang gán nhãn bằng KMeans clustering...")
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(X[:, 0].reshape(-1, 1))  # Sử dụng độ tương đồng cosine để phân cụm
        
        # Gán nhãn 1 cho cụm có độ tương đồng cao hơn
        toaDoTrungTam = kmeans.cluster_centers_.flatten()
        if toaDoTrungTam[1] > toaDoTrungTam[0]:
            df['Label'] = (kmeans.labels_ == 1).astype(int)
        else:
            df['Label'] = (kmeans.labels_ == 0).astype(int)
        
        print(f"Đã hoàn thành xử lý dữ liệu với {len(df)} mẫu.")
        df.to_csv("dataset_labeled.csv", index=False)
        print("Đã lưu dữ liệu đã xử lý vào dataset_labeled.csv")
    except FileNotFoundError:
        print("Không tìm thấy file data_vietnam.csv. Đang thử tải dữ liệu từ Hugging Face...")
        try:
            print("Đang tải dữ liệu từ Hugging Face...")
            try:
                # Thử tải trực tiếp từ Hugging Face
                vi_sts = pd.read_csv("hf://datasets/doanhieung/stsbenchmark-sts-vi/stsbenchmark-sts-vi.tsv", sep="\t")
                df = pd.DataFrame({
                    'Text_A': vi_sts['sentence1'],
                    'Text_B': vi_sts['sentence2']
                })
                print(f"Đã tải xong dữ liệu từ Hugging Face với {len(df)} mẫu.")
            except:
                from datasets import load_dataset
                print("Đang tải từ Hugging Face sử dụng datasets API...")
                dataset = load_dataset("doanhieung/stsbenchmark-sts-vi")
                df = pd.DataFrame({
                    'Text_A': dataset['train']['sentence1'],
                    'Text_B': dataset['train']['sentence2']
                })
                print(f"Đã tải xong dữ liệu từ Hugging Face với {len(df)} mẫu.")
            
            # Lưu dữ liệu gốc
            df.to_csv("data_vietnam.csv", index=False)
            print("Đã lưu dữ liệu gốc vào data_vietnam.csv")
            
            # Tiếp tục xử lý
            print("Đang xử lý văn bản tiếng Việt...")
            
            # Tiền xử lý văn bản
            df['Text_Clean_A'] = df['Text_A'].apply(xuLyNangCao)
            df['Text_Clean_B'] = df['Text_B'].apply(xuLyNangCao)
            
            # Tokenization và loại bỏ stopwords
            df['Text_Token_A'] = df['Text_Clean_A'].apply(ViTokenizer.tokenize)
            df['Text_Token_B'] = df['Text_Clean_B'].apply(ViTokenizer.tokenize)
            
            df['Text_Token_NoStop_A'] = df['Text_Clean_A'].apply(tachTuVaLoaiStopwords)
            df['Text_Token_NoStop_B'] = df['Text_Clean_B'].apply(tachTuVaLoaiStopwords)
            
            # Tính độ tương đồng
            print("Đang tính độ tương đồng giữa các cặp văn bản...")
            X = trich_xuat_dac_trung(df)
            
            # Gán nhãn bằng KMeans
            print("Đang gán nhãn bằng KMeans clustering...")
            kmeans = KMeans(n_clusters=2, random_state=42)
            kmeans.fit(X[:, 0].reshape(-1, 1))  # Sử dụng độ tương đồng cosine để phân cụm
            
            # Gán nhãn 1 cho cụm có độ tương đồng cao hơn
            toaDoTrungTam = kmeans.cluster_centers_.flatten()
            if toaDoTrungTam[1] > toaDoTrungTam[0]:
                df['Label'] = (kmeans.labels_ == 1).astype(int)
            else:
                df['Label'] = (kmeans.labels_ == 0).astype(int)
            
            print(f"Đã hoàn thành xử lý dữ liệu với {len(df)} mẫu.")
            df.to_csv("dataset_labeled.csv", index=False)
            print("Đã lưu dữ liệu đã xử lý vào dataset_labeled.csv")
        except Exception as e:
            print(f"Không thể tải dữ liệu từ Hugging Face: {e}")
            print("Sử dụng dữ liệu giả lập để demo...")
            
            # Tạo dữ liệu giả lập mở rộng nếu không có file
            data = {
                'Text_A': [
                    "hôm nay trời đẹp quá",
                    "tôi thích ăn phở",
                    "học đại học rất quan trọng",
                    "việt nam là một đất nước xinh đẹp",
                    "ngôn ngữ lập trình python rất phổ biến",
                    "hà nội là thủ đô của việt nam",
                    "chào bạn rất vui được gặp bạn",
                    "sách là nguồn tri thức vô tận",
                    "thể thao giúp cơ thể khỏe mạnh",
                    "ai thích ăn cơm không",
                    # Thêm các mẫu mới với độ tương đồng trung bình
                    "Phở là món ăn truyền thống của Việt Nam",
                    "Trí tuệ nhân tạo đang thay đổi thế giới",
                    "Sách là kho tàng tri thức vô giá",
                    "Học tập suốt đời là chìa khóa thành công",
                    "Bảo vệ môi trường là trách nhiệm của mọi người"
                ],
                'Text_B': [
                    "trời đẹp quá",
                    "tôi thích ăn bún bò",
                    "việc học đại học rất quan trọng",
                    "việt nam đất nước xinh đẹp",
                    "python là ngôn ngữ lập trình thông dụng",
                    "hà nội thủ đô việt nam",
                    "chào bạn vui gặp bạn quá",
                    "sách là kho tàng tri thức vô giá",
                    "thể thao giúp tăng cường sức khỏe",
                    "tôi thích ăn cá kho",
                    # Văn bản B cho các mẫu mới
                    "Phở là đặc sản nổi tiếng của ẩm thực Việt Nam",
                    "AI và ML đang phát triển nhanh chóng trong thời đại công nghệ",
                    "Đọc sách là cách hiệu quả để mở rộng kiến thức",
                    "Học hỏi không ngừng giúp phát triển bản thân",
                    "Chúng ta cần có ý thức bảo vệ môi trường sống"
                ]
            }
            df = pd.DataFrame(data)
            
            print("Đang xử lý văn bản tiếng Việt...")
            
            # Tiền xử lý văn bản
            df['Text_Clean_A'] = df['Text_A'].apply(xuLyNangCao)
            df['Text_Clean_B'] = df['Text_B'].apply(xuLyNangCao)
            
            # Tokenization và loại bỏ stopwords
            df['Text_Token_A'] = df['Text_Clean_A'].apply(ViTokenizer.tokenize)
            df['Text_Token_B'] = df['Text_Clean_B'].apply(ViTokenizer.tokenize)
            
            df['Text_Token_NoStop_A'] = df['Text_Clean_A'].apply(tachTuVaLoaiStopwords)
            df['Text_Token_NoStop_B'] = df['Text_Clean_B'].apply(tachTuVaLoaiStopwords)
            
            # Tính độ tương đồng
            print("Đang tính độ tương đồng giữa các cặp văn bản...")
            X = trich_xuat_dac_trung(df)
            
            # Gán nhãn bằng KMeans
            print("Đang gán nhãn bằng KMeans clustering...")
            kmeans = KMeans(n_clusters=2, random_state=42)
            kmeans.fit(X[:, 0].reshape(-1, 1))  # Sử dụng độ tương đồng cosine để phân cụm
            
            # Gán nhãn 1 cho cụm có độ tương đồng cao hơn
            toaDoTrungTam = kmeans.cluster_centers_.flatten()
            if toaDoTrungTam[1] > toaDoTrungTam[0]:
                df['Label'] = (kmeans.labels_ == 1).astype(int)
            else:
                df['Label'] = (kmeans.labels_ == 0).astype(int)
            
            print(f"Đã hoàn thành xử lý dữ liệu với {len(df)} mẫu.")
            df.to_csv("dataset_labeled.csv", index=False)
            print("Đã lưu dữ liệu đã xử lý vào dataset_labeled.csv")

# 7. Nếu df không có cột Text_Token_NoStop_A, thực hiện xử lý
if 'Text_Token_NoStop_A' not in df.columns:
    print("Đang xử lý văn bản từ dữ liệu đã tải...")
    # Tiền xử lý văn bản
    df['Text_Clean_A'] = df['Text_A'].apply(xuLyNangCao)
    df['Text_Clean_B'] = df['Text_B'].apply(xuLyNangCao)
    
    # Tokenization và loại bỏ stopwords
    df['Text_Token_A'] = df['Text_Clean_A'].apply(ViTokenizer.tokenize)
    df['Text_Token_B'] = df['Text_Clean_B'].apply(ViTokenizer.tokenize)
    
    df['Text_Token_NoStop_A'] = df['Text_Clean_A'].apply(tachTuVaLoaiStopwords)
    df['Text_Token_NoStop_B'] = df['Text_Clean_B'].apply(tachTuVaLoaiStopwords)

# 8. Trích xuất đặc trưng từ dữ liệu
print("Đang trích xuất đặc trưng...")
X = trich_xuat_dac_trung(df)
y = df['Label'].values

# 9. Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Chia dữ liệu thành: {len(X_train)} mẫu huấn luyện và {len(X_test)} mẫu kiểm tra.")

# 10. Huấn luyện mô hình KNN với trọng số khoảng cách
print("Đang huấn luyện mô hình KNN...")
knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn_model.fit(X_train, y_train)

# 11. Huấn luyện mô hình Naive Bayes
print("Đang huấn luyện mô hình Naive Bayes...")
nb_model = GaussianNB(var_smoothing=1e-8)  # Tăng thông số làm mịn
nb_model.fit(X_train, y_train)

# 12. Huấn luyện mô hình SVM
print("Đang huấn luyện mô hình SVM...")
svm_model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# 13. Đánh giá mô hình trên tập kiểm tra
print("\nĐánh giá hiệu suất trên tập kiểm tra:")
for name, model in [("KNN", knn_model), ("Naive Bayes", nb_model), ("SVM", svm_model)]:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name}: Accuracy = {accuracy:.4f}")
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix [{name}]:")
    print(f"[[{cm[0][0]}, {cm[0][1]}],")
    print(f" [{cm[1][0]}, {cm[1][1]}]]")

# 14. Lưu mô hình
print("\nĐang lưu các mô hình...")

# Lưu mô hình KNN
with open(os.path.join(MODEL_DIR, 'knn_model.pkl'), 'wb') as f:
    pickle.dump(knn_model, f)

# Lưu mô hình Naive Bayes
with open(os.path.join(MODEL_DIR, 'nb_model.pkl'), 'wb') as f:
    pickle.dump(nb_model, f)

# Lưu mô hình SVM
with open(os.path.join(MODEL_DIR, 'svm_model.pkl'), 'wb') as f:
    pickle.dump(svm_model, f)

# 15. Lưu bộ xử lý văn bản
print("Đang lưu các hàm xử lý văn bản...")
text_processor = {
    'xuLyNangCao': xuLyNangCao,
    'tachTuVaLoaiStopwords': tachTuVaLoaiStopwords,
    'tinh_do_tuong_dong': tinh_do_tuong_dong,
    'danhSachStopwords': danhSachStopwords,
    'tuDienVietTat': tuDienVietTat
}

with open(os.path.join(MODEL_DIR, 'text_processor.pkl'), 'wb') as f:
    pickle.dump(text_processor, f)

print("Quá trình huấn luyện và lưu mô hình hoàn tất.")
print(f"Các mô hình đã được lưu vào thư mục {MODEL_DIR}.")
print("\nBạn có thể chạy ứng dụng Flask bằng lệnh: python app.py") 