from flask import Flask, render_template, request, jsonify
import re
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pyvi import ViTokenizer
from underthesea import text_normalize
import tempfile
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Cấu hình upload
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max
ALLOWED_EXTENSIONS = {'txt', 'doc', 'docx', 'pdf'}

# Đường dẫn đến các mô hình đã lưu
MODEL_DIR = "models"

# Danh sách stopwords và từ điển viết tắt
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

# Kiểm tra phần mở rộng file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Tải các mô hình
def load_models():
    models = {}
    try:
        # Tải các hàm xử lý văn bản
        try:
            with open(os.path.join(MODEL_DIR, 'text_processor.pkl'), 'rb') as f:
                text_processor = pickle.load(f)
                global danhSachStopwords, tuDienVietTat
                danhSachStopwords = text_processor['danhSachStopwords']
                tuDienVietTat = text_processor['tuDienVietTat']
        except:
            print("Không tìm thấy file text_processor.pkl, sử dụng các giá trị mặc định")
        
        # Tải mô hình KNN từ thư viện scikit-learn
        with open(os.path.join(MODEL_DIR, 'knn_model.pkl'), 'rb') as f:
            models['knn'] = pickle.load(f)
        
        # Tải mô hình Naive Bayes từ thư viện scikit-learn
        with open(os.path.join(MODEL_DIR, 'nb_model.pkl'), 'rb') as f:
            models['naive_bayes'] = pickle.load(f)
        
        # Tải mô hình SVM từ thư viện scikit-learn
        try:
            with open(os.path.join(MODEL_DIR, 'svm_model.pkl'), 'rb') as f:
                models['svm'] = pickle.load(f)
                print("Đã tải mô hình SVM thành công")
        except Exception as e:
            # Nếu không tìm thấy mô hình SVM, thử tìm mô hình Decision Tree
            print(f"Không tìm thấy mô hình SVM: {e}")
            try:
                with open(os.path.join(MODEL_DIR, 'dt_model.pkl'), 'rb') as f:
                    models['decision_tree'] = pickle.load(f)
                    print("Đã tải mô hình Decision Tree thành công")
            except:
                print("Không tìm thấy cả hai mô hình SVM và Decision Tree")
        
        return models
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return None

# Các hàm tiền xử lý văn bản
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

# Tính các đặc trưng tương đồng
def tinh_do_tuong_dong(text_a, text_b):
    # 1. Cosine similarity với TF-IDF
    tf_idf = TfidfVectorizer(ngram_range=(1, 2))
    tf_idf_matrix = tf_idf.fit_transform([text_a, text_b])
    cosine = cosine_similarity(tf_idf_matrix[0:1], tf_idf_matrix[1:2])[0][0]
    
    # 2. Jaccard similarity
    set_a = set(text_a.split())
    set_b = set(text_b.split())
    
    union = set_a.union(set_b)
    if len(union) > 0:
        jaccard = len(set_a.intersection(set_b)) / len(union)
    else:
        jaccard = 0
    
    # 3. Tỷ lệ từ chung
    min_len_set = min(len(set_a), len(set_b))
    if min_len_set > 0:
        common_ratio = len(set_a.intersection(set_b)) / min_len_set
    else:
        common_ratio = 0
    
    # 4. Tỷ lệ độ dài
    max_len_text = max(len(text_a), len(text_b))
    if max_len_text > 0:
        len_ratio = min(len(text_a), len(text_b)) / max_len_text
    else:
        len_ratio = 0
    
    return [cosine, jaccard, common_ratio, len_ratio]

# Dự đoán bằng các mô hình
def predict_duplicate(models, features):
    if models is None:
        return {"error": "Không thể tải mô hình"}
    
    features_array = np.array(features).reshape(1, -1)
    results = {}
    
    # Dự đoán với mỗi mô hình
    for name, model in models.items():
        prediction = int(model.predict(features_array)[0])
        probability = model.predict_proba(features_array)[0][1] if hasattr(model, 'predict_proba') else None
        
        results[name] = {
            "prediction": prediction,
            "probability": float(probability) if probability is not None else None
        }
    
    return results

# Trích xuất văn bản từ file
def extract_text_from_file(file):
    filename = secure_filename(file.filename)
    file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    
    # Lưu file vào thư mục tạm thời
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as temp:
        file.save(temp.name)
        temp_path = temp.name
    
    try:
        extracted_text = ""
        
        # Xử lý theo loại file
        if file_ext == 'txt':
            # Đọc file text
            with open(temp_path, 'r', encoding='utf-8', errors='ignore') as f:
                extracted_text = f.read()
        
        elif file_ext in ['doc', 'docx']:
            # Sử dụng thư viện python-docx cho file Word
            try:
                import docx
                doc = docx.Document(temp_path)
                extracted_text = "\n".join([para.text for para in doc.paragraphs])
            except ImportError:
                return "Lỗi: Không thể xử lý file Word. Thư viện python-docx không được cài đặt."
        
        elif file_ext == 'pdf':
            # Sử dụng thư viện PyPDF2 cho file PDF
            try:
                import PyPDF2
                with open(temp_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    extracted_text = ""
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        extracted_text += page.extract_text() + "\n"
            except ImportError:
                return "Lỗi: Không thể xử lý file PDF. Thư viện PyPDF2 không được cài đặt."
        
        # Xóa file tạm sau khi xử lý
        os.unlink(temp_path)
        return extracted_text
    
    except Exception as e:
        # Đảm bảo file tạm được xóa trong trường hợp lỗi
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        return f"Lỗi khi xử lý file: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_duplicate', methods=['POST'])
def check_duplicate():
    # Lấy dữ liệu từ form
    text_a = request.form.get('text_a', '')
    text_b = request.form.get('text_b', '')
    
    if not text_a or not text_b:
        return jsonify({
            "error": "Vui lòng nhập đủ hai đoạn văn bản!"
        })
    
    # Tiền xử lý văn bản nâng cao
    text_a_processed = xuLyNangCao(text_a)
    text_b_processed = xuLyNangCao(text_b)
    
    # Tokenize tiếng Việt và loại bỏ stopwords
    text_a_tokenized = tachTuVaLoaiStopwords(text_a_processed)
    text_b_tokenized = tachTuVaLoaiStopwords(text_b_processed)
    
    # Tính các đặc trưng
    features = tinh_do_tuong_dong(text_a_tokenized, text_b_tokenized)
    
    # Tải mô hình
    models = load_models()
    
    # Dự đoán
    predictions = predict_duplicate(models, features)
    
    # Phân tích từ chung
    set_a = set(text_a_tokenized.split())
    set_b = set(text_b_tokenized.split())
    common_words = set_a.intersection(set_b)
    
    # Trả về kết quả
    return jsonify({
        "text_a": text_a,
        "text_b": text_b,
        "features": {
            "cosine": features[0],
            "jaccard": features[1],
            "common_ratio": features[2],
            "len_ratio": features[3]
        },
        "common_words": list(common_words),
        "predictions": predictions
    })

@app.route('/check_duplicate_files', methods=['POST'])
def check_duplicate_files():
    # Kiểm tra xem có file được gửi lên không
    if 'file_a' not in request.files or 'file_b' not in request.files:
        return jsonify({
            "error": "Vui lòng tải lên đủ hai file văn bản!"
        })
    
    file_a = request.files['file_a']
    file_b = request.files['file_b']
    
    # Kiểm tra tên file
    if file_a.filename == '' or file_b.filename == '':
        return jsonify({
            "error": "Vui lòng chọn đủ hai file văn bản!"
        })
    
    # Kiểm tra định dạng file
    if not allowed_file(file_a.filename) or not allowed_file(file_b.filename):
        return jsonify({
            "error": "Định dạng file không được hỗ trợ. Hãy sử dụng TXT, DOC, DOCX hoặc PDF."
        })
    
    # Trích xuất văn bản từ file
    text_a = extract_text_from_file(file_a)
    text_b = extract_text_from_file(file_b)
    
    # Kiểm tra nếu có lỗi trong quá trình trích xuất
    if isinstance(text_a, str) and text_a.startswith("Lỗi:"):
        return jsonify({"error": text_a})
    if isinstance(text_b, str) and text_b.startswith("Lỗi:"):
        return jsonify({"error": text_b})
    
    # Tiền xử lý văn bản nâng cao
    text_a_processed = xuLyNangCao(text_a)
    text_b_processed = xuLyNangCao(text_b)
    
    # Tokenize tiếng Việt và loại bỏ stopwords
    text_a_tokenized = tachTuVaLoaiStopwords(text_a_processed)
    text_b_tokenized = tachTuVaLoaiStopwords(text_b_processed)
    
    # Tính các đặc trưng
    features = tinh_do_tuong_dong(text_a_tokenized, text_b_tokenized)
    
    # Tải mô hình
    models = load_models()
    
    # Dự đoán
    predictions = predict_duplicate(models, features)
    
    # Phân tích từ chung
    set_a = set(text_a_tokenized.split())
    set_b = set(text_b_tokenized.split())
    common_words = set_a.intersection(set_b)
    
    # Trả về kết quả
    return jsonify({
        "file_a": file_a.filename,
        "file_b": file_b.filename,
        "text_a": text_a[:300] + "..." if len(text_a) > 300 else text_a,
        "text_b": text_b[:300] + "..." if len(text_b) > 300 else text_b,
        "features": {
            "cosine": features[0],
            "jaccard": features[1],
            "common_ratio": features[2],
            "len_ratio": features[3]
        },
        "common_words": list(common_words),
        "predictions": predictions
    })

# Tạo thư mục models nếu chưa tồn tại
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

if __name__ == '__main__':
    app.run(debug=True) 