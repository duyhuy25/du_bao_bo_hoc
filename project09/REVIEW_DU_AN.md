# Review dự án cảnh báo sớm nguy cơ bỏ học

Ngày review: 10/10/2026. Phạm vi: mã nguồn, dữ liệu CSV, nội dung và output đã lưu của notebook, các báo cáo trong thư mục `project09`.

Review này giải thích phiên bản hiện có; chưa sửa mã và chưa chạy lại huấn luyện. Các số liệu bên dưới lấy từ dữ liệu và kết quả đã lưu, không phải kết quả của một lần chạy mới. Trong phiên review, lệnh `python` của môi trường công cụ không chạy được nên chưa kiểm chứng lại ứng dụng bằng thực thi.

## 1. Bài toán và cách hệ thống hoạt động

Dự án dự đoán trạng thái của sinh viên từ thông tin đầu vào, với ba nhãn:

| Nhãn | Ý nghĩa |
|---|---|
| `Dropout` | Bỏ học |
| `Enrolled` | Đang theo học |
| `Graduate` | Đã tốt nghiệp |

Mục tiêu là cảnh báo ngay khi nhập học. Vì vậy, các kết quả học kỳ 1 và học kỳ 2 bị loại khỏi đầu vào: tại thời điểm dự đoán, nhà trường chưa biết những kết quả này.

Hệ thống có hai luồng tách biệt:

```text
HUẤN LUYỆN
data/raw/data.csv
  → src/data.py: đọc dữ liệu, chia train/validation/test
  → src/features.py: bỏ biến tương lai, chuẩn hóa, mã hóa
  → src/train.py: huấn luyện Random Forest
  → models/dropout_model.pkl: lưu cả pipeline

DỰ ĐOÁN TRÊN WEB
app/templates/index.html: nhập thông tin
  → POST /predict
  → app/main.py: chuyển JSON thành DataFrame
  → pipeline đã lưu: tiền xử lý và dự đoán xác suất
  → phân mức ưu tiên theo xác suất Dropout
  → giao diện hiển thị kết quả
```

Web không train lại khi người dùng bấm dự đoán. Nó nạp mô hình đã train và sử dụng mô hình đó cho các yêu cầu mới.

## 2. Giải thích từng file

### `data/raw/data.csv` — dữ liệu đầu vào

File hiện có 4.424 dòng và 37 cột: 36 cột đặc trưng cùng cột nhãn `Target`.

| Nhãn | Số dòng |
|---|---:|
| Graduate | 2.209 |
| Dropout | 1.421 |
| Enrolled | 794 |

Số lượng giữa các lớp không bằng nhau. Nếu luôn đoán lớp đông nhất, mô hình vẫn đạt gần 50% accuracy, dù không nhận diện được sinh viên bỏ học. Vì vậy bài cần thêm macro-F1 và recall của lớp Dropout.

Output notebook đã lưu báo cáo không có giá trị thiếu và không có dòng trùng. Mã huấn luyện chưa có bước điền giá trị thiếu; nếu thay dữ liệu mới, cần kiểm tra lại điều này.

### `src/data.py` — đọc và chia dữ liệu

Hàm `load_data()` xác định thư mục gốc dự án dựa trên vị trí của chính file Python, ghép với `data/raw/data.csv`, kiểm tra file tồn tại rồi đọc bằng `pd.read_csv()`.

Hàm `get_train_val_test_splits(df)` thực hiện:

1. Tạo `X` bằng cách bỏ cột `Target`; tạo `y` từ cột `Target`.
2. Tách 20% dữ liệu làm test.
3. Tách 25% của 80% còn lại làm validation: `0,25 × 0,80 = 0,20`.
4. Trả về `X_train, X_val, X_test, y_train, y_val, y_test`.

Kích thước trong lần chạy notebook đã lưu:

| Tập | Số dòng | Vai trò |
|---|---:|---|
| Train | 2.654 | Học tham số tiền xử lý và mô hình |
| Validation | 885 | So sánh, lựa chọn mô hình hoặc ngưỡng |
| Test | 885 | Đánh giá cuối sau khi chốt lựa chọn |

`stratify=y` giúp giữ gần giống tỷ lệ ba lớp giữa các tập. `random_state=42` cố định phép chia khi dữ liệu và điều kiện chạy giữ nguyên. Nếu thay thứ tự hoặc nội dung dữ liệu, không nên mặc định rằng các tập vẫn giống trước.

File này chỉ đọc và chia dữ liệu; việc bỏ biến học kỳ nằm trong `features.py`.

### `src/features.py` — tiền xử lý

File có hai thành phần chính.

**`DropFutureFeatures`** là transformer tùy chỉnh tương thích với Pipeline của scikit-learn. `fit()` chỉ trả về chính đối tượng vì bước này không cần học tham số. `transform()` sao chép dữ liệu, tìm các tên cột chứa `1st sem` hoặc `2nd sem`, rồi loại chúng.

Theo output notebook, số cột giảm từ 36 xuống 24. Việc loại 12 cột học kỳ nhằm tránh dùng thông tin tương lai để dự đoán thời điểm nhập học.

**`build_preprocessor(X_train)`** tạo `ColumnTransformer` với hai nhánh:

| Nhánh | Các cột | Cách xử lý |
|---|---|---|
| Số liên tục | `Previous qualification (grade)`, `Admission grade`, `Age at enrollment`, `Unemployment rate`, `Inflation rate`, `GDP` | `StandardScaler` |
| Phân loại | 18 cột còn lại sau khi bỏ biến học kỳ | `OneHotEncoder(handle_unknown='ignore')` |

StandardScaler đưa một giá trị về dạng gần như `z = (x − trung bình train) / độ lệch chuẩn train`. Tham số được học trên train và dùng lại cho validation, test, dữ liệu web.

One-hot biến mã nhóm thành các cột chỉ báo. Ví dụ một cột ngành có ba mã có thể trở thành ba cột thể hiện sinh viên thuộc ngành nào. Mã số ngành được coi là nhóm, không phải đại lượng có quan hệ lớn–nhỏ.

`handle_unknown='ignore'` giúp encoder không báo lỗi khi gặp mã nhóm chưa xuất hiện lúc train; nhóm cột tương ứng được mã hóa toàn số 0. Nó không tự giải quyết lỗi thiếu cột đầu vào.

Trong lần chạy notebook đã lưu, 24 cột được chuyển thành 223 đặc trưng sau encoding. Con số này phụ thuộc các nhóm xuất hiện trong tập train.

StandardScaler thường không cần thiết cho Random Forest, nhưng có ích cho Logistic Regression khi so sánh trong notebook. Pipeline dùng chung giúp các ứng viên nhận cùng quy trình tiền xử lý.

### `src/train.py` — huấn luyện mô hình dùng cho web

Hàm `train_model()` đọc dữ liệu, chia tập, tạo bộ tiền xử lý và ghép ba bước:

```python
Pipeline([
    ('dropper', DropFutureFeatures()),
    ('preprocessor', build_preprocessor(X_train)),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    ))
])
```

Lệnh thực sự huấn luyện là:

```python
pipeline.fit(X_train, y_train)
```

Khi gọi `fit`, pipeline lần lượt bỏ cột học kỳ, học scaler và encoder từ train, chuyển đổi dữ liệu rồi huấn luyện bộ phân loại.

Random Forest kết hợp nhiều cây quyết định. Mỗi cây học các quy tắc chia dữ liệu; sự ngẫu nhiên trong lấy mẫu và chọn đặc trưng giúp các cây khác nhau. Khi dự đoán xác suất, rừng tổng hợp xác suất từ các cây.

Ý nghĩa các cấu hình được đặt trong file:

| Tham số | Ý nghĩa |
|---|---|
| `n_estimators=100` | Dùng 100 cây |
| `random_state=42` | Cố định nguồn ngẫu nhiên của mô hình |
| `class_weight='balanced'` | Tăng trọng số tương đối cho lớp ít mẫu dựa trên tần suất lớp |

`balanced` không tạo thêm dữ liệu và không bảo đảm recall Dropout tăng. Cần đo kết quả thực tế trên validation.

Cuối cùng, `joblib.dump()` lưu toàn bộ pipeline vào `models/dropout_model.pkl`, bao gồm transformer, scaler, encoder và Random Forest đã học. Vì vậy web không cần tự viết lại các bước chuẩn hóa.

Lưu ý: file này chỉ train một cấu hình Random Forest cố định; chưa tìm siêu tham số, chưa tự chọn mô hình tốt nhất, chưa đánh giá validation/test. Chạy lại sẽ ghi đè `dropout_model.pkl`.

### `src/evaluate.py` — đánh giá trên validation

File đọc lại dữ liệu và chia theo cùng cấu hình, nạp `models/dropout_model.pkl`, dự đoán trên `X_val`, rồi in `classification_report`.

Các chỉ số cần hiểu:

| Chỉ số | Câu hỏi nó trả lời |
|---|---|
| Accuracy | Trong tất cả sinh viên, dự đoán đúng bao nhiêu phần? |
| Precision Dropout | Trong những người được dự đoán bỏ học, bao nhiêu người thực sự bỏ học? |
| Recall Dropout | Trong những người thực sự bỏ học, mô hình tìm ra được bao nhiêu người? |
| F1 | Mức cân bằng giữa precision và recall |
| Macro-F1 | Trung bình F1 của ba lớp, mỗi lớp có trọng số bằng nhau |

Ví dụ recall Dropout = 0,64 nghĩa là mô hình nhận diện được khoảng 64% trường hợp bỏ học trong tập được đánh giá, theo quy tắc phân lớp đang dùng.

File cũng tạo confusion matrix: hàng là nhãn thật, cột là nhãn dự đoán. Ô trên đường chéo là dự đoán đúng; một sinh viên thật sự Dropout nhưng được đoán Graduate là trường hợp bỏ sót.

Ảnh được ghi vào `reports/figures/confusion_matrix.png`. Dù hàm chia tập trả cả test, mã này chỉ đánh giá validation. Đánh giá test nằm trong notebook lớn.

### `app/main.py` — API phục vụ dự đoán

Ứng dụng FastAPI thiết lập thư mục HTML, thư mục static và nạp `models/dropout_model.pkl` khi khởi động.

Hai endpoint được khai báo trong mã:

| Endpoint | Chức năng |
|---|---|
| `GET /` | Trả trang `index.html` |
| `POST /predict` | Nhận dữ liệu một sinh viên và trả kết quả |

Trong `/predict`, dictionary đầu vào được chuyển thành DataFrame một dòng bằng `pd.DataFrame([data])`. Pipeline chạy `predict_proba()`, sau đó ghép các xác suất với `pipeline.classes_` để tránh nhầm thứ tự lớp.

API làm tròn xác suất đến bốn chữ số thập phân, lấy xác suất Dropout rồi áp dụng quy tắc:

| Xác suất Dropout sau làm tròn | Mức trả về |
|---|---|
| `p > 0.6` | `Critical` |
| `0.4 < p <= 0.6` | `High` |
| `p <= 0.4` | `Normal` |

Đúng 0,60 thuộc High; đúng 0,40 thuộc Normal. Đây là các ngưỡng được viết cố định trong code, chưa có bước tối ưu ngưỡng trong quy trình hiện có.

Kết quả JSON gồm `prediction_probabilities`, `support_priority`, `suggestion`. Các câu gợi ý là nội dung cố định tương ứng với mức ưu tiên, không phải văn bản do một mô hình ngôn ngữ tạo ra.

Hiện đầu vào được khai báo là `dict`, chưa có schema kiểm tra đủ 24 trường, phạm vi và kiểu dữ liệu cụ thể. `BaseModel` được import nhưng chưa sử dụng. Lỗi được trả dưới dạng `{"error": "..."}`, chưa đặt mã HTTP lỗi tương ứng.

### `app/templates/index.html` — giao diện và JavaScript

File này chứa HTML, CSS nội tuyến và JavaScript. Có hai cách nhập: form thủ công và JSON.

Luồng form:

1. Người dùng sửa 11 trường đang hiển thị.
2. JavaScript lặp qua 24 khóa của `sampleJson`.
3. Trường có ô nhập lấy giá trị người dùng; trường không có ô nhập lấy giá trị mẫu.
4. Quy đổi điểm cấp 3 bằng cách nhân 20; điểm xét tuyển bằng cách chia 40 rồi nhân 200.
5. Gửi JSON đến `/predict` bằng `fetch()`.
6. Hiển thị xác suất, màu cảnh báo và gợi ý từ API.

Như vậy mô hình vẫn nhận 24 trường, nhưng 13 trường được điền bằng giá trị mẫu khi dùng form. Dòng mô tả “điền đủ 24 trường” chưa phản ánh đúng giao diện hiện tại.

Tab JSON gửi trực tiếp dữ liệu đã nhập, không thực hiện phép quy đổi điểm của form. Khi dùng tab này cần cung cấp điểm theo thang đầu vào của mô hình.

Một tên cột cần chú ý là `Daytime/evening attendance\t`: ký tự tab cuối tên thuộc schema hiện tại. Tên `Nacionality` cũng cần khớp nguyên văn dữ liệu. Đổi tên chỉ ở frontend có thể làm pipeline thiếu cột.

Phép đổi thang điểm đang là giả định của giao diện. Chỉ đổi thang số chưa chứng minh dữ liệu của hai hệ thống giáo dục tương đương để dự đoán.

### `app/static/style.css` — CSS riêng

File chứa một số định dạng cơ bản cho trang. Tuy nhiên, `index.html` hiện dùng thẻ `<style>` bên trong và không có thẻ liên kết đến `/static/style.css`. Vì vậy sửa file CSS riêng này chưa làm thay đổi giao diện hiện tại.

### `run.py` — khởi động web

File gọi Uvicorn để chạy `app.main:app` tại `127.0.0.1:8000`, có `reload=True` để tự tải lại khi sửa mã trong lúc phát triển.

Chạy `python run.py` chỉ mở server; không tự train mô hình. Nếu vừa thay file mô hình, nên khởi động lại server để nạp bản mới.

### `requirements.txt` — thư viện phụ thuộc

Danh sách hiện có: pandas, numpy, scikit-learn, matplotlib, seaborn, fastapi, uvicorn, jupyter.

Mã còn dùng `joblib`, `jinja2`, `nbformat`, `nbclient`. Một số có thể được cài gián tiếp qua thư viện khác, nhưng nên khai báo trực tiếp các phụ thuộc dự án sử dụng. Đặc biệt, template của web cần `jinja2`, hiện chưa có trong danh sách.

File chưa khóa phiên bản, nên cài mới có thể cho môi trường khác lần train cũ. Output notebook ghi nhận Python 3.11.9 và scikit-learn 1.9.1; đây là thông tin từ output lưu sẵn, chưa được xác minh bằng lần chạy mới trong review này.

### `thong_tin_du_lieu.ipynb` — xem cấu trúc dữ liệu

Notebook đọc CSV rồi chạy `df.info()` để xem tên cột, kiểu dữ liệu và số giá trị không rỗng. Nó không huấn luyện mô hình.

Đường dẫn đọc CSV đang cố định là `D:\học máy\project09\data\raw\data.csv`. Trên máy hoặc thư mục khác cần đổi đường dẫn, chẳng hạn `pd.read_csv('data/raw/data.csv')` khi kernel chạy tại thư mục gốc dự án.

### `xem_du_lieu.ipynb` — xem 11 trường trên form

Notebook chọn 11 cột tương ứng các trường người dùng sửa trên web, hiển thị các dòng đầu và thông tin kiểu dữ liệu. Đây là file khám phá dữ liệu, không phải một pipeline train chỉ dùng 11 đặc trưng.

File cũng dùng đường dẫn tuyệt đối giống notebook phía trên và cần chỉnh khi chuyển máy.

### `train_eval_dropout.ipynb` — quy trình thử nghiệm đầy đủ

Notebook này thực hiện nhiều việc hơn `src/train.py`:

1. Ghi phiên bản môi trường và hash dữ liệu.
2. Kiểm tra kích thước, nhãn, dữ liệu thiếu và trùng.
3. Chia train/validation/test; kiểm tra các tập không trùng chỉ số dòng.
4. Loại cột học kỳ và thiết lập tiền xử lý.
5. Train bốn ứng viên: Dummy, Logistic Regression balanced, RF không trọng số và RF balanced.
6. Đánh giá train cùng validation để so sánh và xem dấu hiệu overfitting.
7. Chọn theo macro-F1 validation cao nhất; nếu hòa thì xét recall Dropout rồi accuracy.
8. Đánh giá test cho ứng viên được chọn và RF của dự án nếu đó là hai mô hình khác nhau.
9. Nạp artifact cũ để đối chiếu kết quả với RF vừa train.
10. Ghi bảng CSV và lưu RF vừa train thành `dropout_model_retrained.pkl`.

Điểm quan trọng: notebook chọn Logistic Regression trong kết quả đã lưu, nhưng file được xuất ở bước cuối vẫn là RF balanced. Notebook không thay `dropout_model.pkl`, và web không tự chuyển sang Logistic Regression.

Tên `dropout_model` chỉ có nghĩa mô hình dự đoán bỏ học, không phải kỹ thuật dropout trong mạng neural.

### `build_train_eval_notebook.py` — sinh và thực thi notebook

Script dùng `nbformat` để tạo các cell rồi dùng `NotebookClient` để chạy và lưu notebook kèm output. Khi chạy, nó thực hiện cả huấn luyện và xuất báo cáo, không chỉ tạo một notebook rỗng.

Biến `root=Path('project09').resolve()` giả định thư mục làm việc có thư mục con `project09`. Với bố cục máy hiện tại, cần chạy từ `D:\học máy`:

```powershell
python project09/build_train_eval_notebook.py
```

Repo GitHub hiện đặt nội dung dự án ngay ở gốc. Nếu chạy script từ gốc repo đó, đường dẫn trên sẽ trỏ sai sang thư mục con `project09`. Cần sửa cách tìm root trước khi dùng script tại bản clone; cách đơn giản là dùng thư mục chứa script: `Path(__file__).resolve().parent`.

Script ghi lại `train_eval_dropout.ipynb` và các kết quả liên quan. Nếu muốn giữ output cũ để đối chiếu, cần lưu phiên bản trước khi chạy lại.

### Hai file trong `models/`

| File | Nguồn tạo và nơi dùng |
|---|---|
| `dropout_model.pkl` | Được `src/train.py` ghi; web và `src/evaluate.py` nạp |
| `dropout_model_retrained.pkl` | Được notebook lưu từ RF balanced vừa huấn luyện; web hiện không nạp |

Các file này chứa pipeline đã học. Khi nạp, môi trường cần có mã transformer `src.features.DropFutureFeatures` và các thư viện tương thích. Không thể xem `.pkl` như một file cấu hình văn bản độc lập với mã nguồn.

### Các file trong `reports/`

| File | Nội dung |
|---|---|
| `figures/confusion_matrix.png` | Ma trận nhầm lẫn validation do `src/evaluate.py` xuất |
| `train_eval/train_validation_metrics.csv` | Chỉ số trên train và validation của bốn ứng viên |
| `train_eval/test_metrics.csv` | Chỉ số test của Logistic balanced và RF balanced |
| `train_eval/existing_artifact_metrics.csv` | Chỉ số validation/test khi nạp artifact cũ |

Các ảnh confusion matrix test của notebook được lưu trong output notebook; mã notebook không xuất chúng thành các PNG riêng ở thư mục báo cáo.

### `README.md`, `phan_cong_nhom_9.md` và `.gitignore`

`README.md` giới thiệu mục tiêu, cấu trúc và cách chạy. Với cách import `from src...` hiện tại, nên chạy module từ gốc dự án bằng `python -m src.train` và `python -m src.evaluate` thay cho lệnh chạy file trực tiếp ghi trong README.

`phan_cong_nhom_9.md` là tài liệu phân công và nhật ký. Một số mô tả chưa khớp triển khai: tài liệu nói Logistic Regression là mô hình chính, endpoint `/api/support-priority`, web ba màn hình và các thử nghiệm mở rộng. Mã hiện tại phục vụ RF qua `/predict`; chưa thấy đầy đủ bằng chứng thực thi các mục thử nghiệm được nêu. Cần chỉnh tài liệu hoặc bổ sung triển khai, không nên dùng dấu “100%” trong tài liệu làm bằng chứng rằng chức năng đã có.

`.gitignore` được thêm ở gốc repo khi đăng bài để bỏ qua cache Python, môi trường ảo, checkpoint notebook và file môi trường. File này không tham gia train hay dự đoán. Bản nguồn `project09` tại máy chưa được thêm file này trong quá trình đăng bằng worktree tạm.

Các file `__pycache__/*.pyc` trên máy là bytecode do Python sinh tự động; không phải mã nguồn cần chỉnh sửa và không được đưa lên GitHub.

## 3. Cách cài, train, đánh giá và chạy web

### Bước 1 — đứng đúng thư mục

Với bài trên máy hiện tại:

```powershell
cd "D:\học máy\project09"
```

Nếu clone mới từ GitHub:

```powershell
git clone --branch NguyenDuyHuy --single-branch https://github.com/duyhuy25/du_bao_bo_hoc.git
cd du_bao_bo_hoc
```

Chỉ chọn một cách. Thư mục làm việc đúng là nơi có `src`, `app`, `data`, `models` và `requirements.txt`.

### Bước 2 — tạo môi trường và cài thư viện

Các lệnh sau giả định máy đã có Python hoạt động. Gọi trực tiếp Python trong môi trường ảo giúp không cần kích hoạt PowerShell script:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install jinja2 joblib nbformat nbclient
```

Nếu dùng Python Launcher trên Windows, có thể thay lệnh tạo môi trường bằng `py -3 -m venv .venv`. Khi nạp model cũ gặp lỗi khác phiên bản, cần dùng môi trường tương thích hoặc train lại trong môi trường vừa cài.

### Bước 3 — train mô hình của ứng dụng

```powershell
.\.venv\Scripts\python.exe -m src.train
```

Các thông báo dự kiến là đọc dữ liệu, chia tập, xây pipeline, training và lưu thành công. Kết quả là `models/dropout_model.pkl` mới. Lệnh này không chạy notebook, không tự so sánh bốn mô hình và không cập nhật các CSV báo cáo của notebook.

### Bước 4 — đánh giá validation

```powershell
.\.venv\Scripts\python.exe -m src.evaluate
```

Đọc precision, recall, F1 của từng lớp trong terminal và xem `reports/figures/confusion_matrix.png`. Cần quan tâm cả recall Dropout và tỷ lệ cảnh báo sai, không chỉ accuracy.

### Bước 5 — chạy web

```powershell
.\.venv\Scripts\python.exe run.py
```

Mở `http://127.0.0.1:8000`, nhập thông tin rồi bấm dự đoán. Dừng server bằng `Ctrl+C`. Sau khi train và thay model, khởi động lại server để nạp model mới.

### Bước 6 — chạy quy trình so sánh và test

```powershell
.\.venv\Scripts\python.exe -m jupyter notebook
```

Mở `train_eval_dropout.ipynb`, chọn kernel của môi trường vừa cài và chạy các cell từ đầu đến cuối. Theo dõi kết quả validation trước khi đọc test; không đổi lựa chọn chỉ vì một ứng viên có test cao hơn.

Notebook sẽ train bốn ứng viên, ghi lại các CSV trong `reports/train_eval` và cập nhật `models/dropout_model_retrained.pkl`. Web vẫn dùng `dropout_model.pkl`.

## 4. Diễn giải kết quả đã lưu

Kết quả validation trong `train_validation_metrics.csv`:

| Mô hình | Accuracy | Macro-F1 | Recall Dropout | Precision Dropout |
|---|---:|---:|---:|---:|
| Dummy | 49,94% | 0,2221 | 0,00% | 0,00% |
| Logistic balanced | 60,68% | **0,5770** | 57,04% | 65,59% |
| RF unweighted | **65,31%** | 0,5220 | 63,03% | 64,86% |
| RF balanced | 63,28% | 0,5598 | **64,08%** | 61,90% |

Theo tiêu chí macro-F1 validation đã đặt trong notebook, Logistic balanced được chọn. RF unweighted có accuracy cao nhất nhưng không có macro-F1 cao nhất. RF balanced có recall Dropout cao nhất trong bốn ứng viên trên lần chia này, đổi lại precision thấp hơn Logistic balanced.

Kết quả test trong `test_metrics.csv`:

| Mô hình | Accuracy | Macro-F1 | Recall Dropout | Precision Dropout |
|---|---:|---:|---:|---:|
| Logistic balanced | 57,29% | **0,5439** | 59,15% | **65,88%** |
| RF balanced | **59,10%** | 0,5219 | **61,27%** | 58,78% |

Hai RF có accuracy và macro-F1 train bằng 1,0, trong khi validation thấp hơn nhiều. Đây là dấu hiệu đáng xem xét về overfitting, không phải bằng chứng mô hình đã hoàn hảo.

Notebook báo artifact cũ và RF train lại cho cùng dự đoán trên validation/test. Điều này không chứng minh lịch sử huấn luyện của artifact cũ hoặc bảo đảm test chưa từng được dùng với artifact đó.

Các chỉ số trên dùng `predict()` để chọn lớp có xác suất cao nhất. Chúng không trực tiếp đo hiệu quả quy tắc cảnh báo `p > 0.4` hay `p > 0.6` trên web. Muốn đánh giá cảnh báo, cần đo riêng precision/recall và số cảnh báo tại từng ngưỡng.

## 5. Những điểm nên sửa, theo mức ưu tiên

| Ưu tiên | Vấn đề | Hướng xử lý |
|---|---|---|
| Cao | Mô hình notebook chọn khác mô hình web dùng | Xác định tiêu chí phục vụ thực tế; lưu và nạp đúng pipeline được chọn, hoặc giải thích rõ lý do giữ RF |
| Cao | 13 trường web đang lấy giá trị mẫu | Cho nhập đủ trường, lấy dữ liệu thật từ hệ thống hoặc train một mô hình dành riêng cho các trường sẵn có |
| Cao | Chưa kiểm tra schema đầu vào | Khai báo model Pydantic với tên trường, kiểu và phạm vi; trả mã HTTP phù hợp |
| Cao | Giả định quy đổi thang điểm và ánh xạ ngành chưa được kiểm chứng | Đối chiếu từ điển dữ liệu, mô tả rõ giới hạn khi áp dụng cho nhóm sinh viên khác |
| Cao | Tài liệu chưa khớp mã | Sửa tên mô hình, endpoint, hướng dẫn chạy và trạng thái các thử nghiệm |
| Trung bình | Ngưỡng cảnh báo cố định, chưa đánh giá riêng | Chọn ngưỡng trên validation theo chi phí bỏ sót/cảnh báo sai rồi đánh giá test |
| Trung bình | RF có khoảng cách train–validation lớn | Thử giới hạn độ sâu, tăng số mẫu tối thiểu ở lá và đánh giá chéo trên dữ liệu phát triển |
| Trung bình | Đường dẫn notebook/script phụ thuộc máy | Chuyển sang đường dẫn tương đối hoặc tìm root từ vị trí file |
| Trung bình | Phụ thuộc chưa đầy đủ, chưa khóa phiên bản | Bổ sung thư viện trực tiếp và ghi môi trường train cùng model |
| Thấp | CSS riêng chưa được sử dụng | Liên kết stylesheet trong HTML hoặc gom CSS về một nơi |

Ngoài các cột học kỳ đã bỏ, cần xác minh thời điểm ghi nhận những biến như nợ và tình trạng đóng học phí. Nếu chỉ biết chúng sau thời điểm dự đoán, vẫn có nguy cơ rò rỉ thông tin thời gian.

## 6. Cách trình bày bài khi thuyết trình

Có thể diễn giải theo chuỗi sau:

> Bài toán của nhóm là phân loại sinh viên thành Dropout, Enrolled và Graduate để hỗ trợ cảnh báo sớm. Dữ liệu có 4.424 sinh viên. Nhóm chia train, validation, test theo tỷ lệ 60/20/20 và giữ tỷ lệ lớp bằng stratify. Vì dự đoán ở thời điểm nhập học, nhóm loại 12 biến liên quan kết quả học kỳ, còn 24 biến đầu vào. Các biến số được chuẩn hóa và biến phân loại được one-hot; toàn bộ tham số tiền xử lý chỉ học trên train. Ứng dụng hiện dùng Random Forest 100 cây và lưu cả pipeline bằng joblib. Notebook so sánh bốn ứng viên cho thấy Logistic Regression balanced có macro-F1 validation cao nhất, còn RF balanced có recall Dropout cao hơn. Do đó cần giải thích rõ tiêu chí chọn mô hình, không khẳng định RF là tốt nhất ở mọi chỉ số. Web nạp pipeline đã lưu, nhận dữ liệu và trả xác suất cùng mức ưu tiên hỗ trợ.

Khi được hỏi “train ở đâu?”, chỉ vào `pipeline.fit(X_train, y_train)` trong `src/train.py`. Khi được hỏi “vì sao chia ba tập?”, giải thích train để học, validation để lựa chọn và test để đánh giá cuối. Khi được hỏi “web có học thêm từ người dùng không?”, trả lời phiên bản hiện tại chỉ suy luận và không cập nhật mô hình từ các lần nhập form.
