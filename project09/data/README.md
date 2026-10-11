# Hồ Sơ Dữ Liệu (Data Documentation & Dictionary)
Dự án: Cảnh báo sớm nguy cơ bỏ học từ dữ liệu đầu vào (Project 09)

## 1. Nguồn Gốc và Giấy Phép
* **Tên tập dữ liệu:** Predict Students' Dropout and Academic Success
* **Kho lưu trữ chính thức:** UCI Machine Learning Repository
* **URL:** https://archive.ics.uci.edu/dataset/697/predict%2Bstudents%2Bdropout%2Band%2Bacademic%2Bsuccess
* **DOI:** https://doi.org/10.24432/C5MC89
* **Bản quyền / Giấy phép sử dụng:** Creative Commons Attribution 4.0 International (CC BY 4.0).
* **Quy mô:** 4.424 dòng (sinh viên), 36 đặc trưng đầu vào, 1 cột nhãn (`Target`).

## 2. Phân Bố Nhãn Mục Tiêu (Target Distribution)
Bài toán phân loại 3 lớp mất cân bằng:
* **Graduate:** 2.209 sinh viên (~49.9%)
* **Dropout:** 1.421 sinh viên (~32.1%)
* **Enrolled:** 794 sinh viên (~17.9%)

## 3. Mốc Thời Gian & Quy Tắc Chống Rò Rỉ Dữ Liệu (Data Leakage)
Mục tiêu bài toán là **cảnh báo sớm ngay khi sinh viên nhập học**. Do đó:
* **12 biến học kỳ bị LOẠI BỎ hoàn toàn:** Tất cả các biến chứa cụm từ `1st sem` hoặc `2nd sem` (tín chỉ đăng ký, tích lũy, điểm số kỳ 1 & 2) đều là thông tin tương lai chưa tồn tại ở tuần lễ nhập học.
* **24 biến được GIỮ LẠI:** Bao gồm thông tin nhân khẩu học (Tuổi, Giới tính, Tình trạng hôn nhân, Quốc tịch), Hoàn cảnh kinh tế - gia đình (Trình độ & Nghề nghiệp bố mẹ, Nợ học phí, Học bổng), Kết quả học tập trước đại học (Điểm cấp 3, Điểm xét tuyển), và Chỉ số kinh tế vĩ mô (Tỷ lệ thất nghiệp, Lạm phát, GDP).

## 4. Từ Điển Dữ Liệu Rút Gọn (Data Dictionary)
| Tên biến | Kiểu dữ liệu | Vai trò | Thời điểm có sẵn |
| :--- | :--- | :--- | :--- |
| `Marital status` | Categorical | Feature | Lúc nhập học |
| `Application mode` | Categorical | Feature | Lúc nhập học |
| `Application order` | Categorical | Feature | Lúc nhập học |
| `Course` | Categorical | Feature | Lúc nhập học |
| `Daytime/evening attendance` | Categorical | Feature | Lúc nhập học |
| `Previous qualification` | Categorical | Feature | Lúc nhập học |
| `Previous qualification (grade)` | Continuous (Float) | Feature | Lúc nhập học |
| `Nacionality` | Categorical | Feature | Lúc nhập học |
| `Mother's qualification` | Categorical | Feature | Lúc nhập học |
| `Father's qualification` | Categorical | Feature | Lúc nhập học |
| `Mother's occupation` | Categorical | Feature | Lúc nhập học |
| `Father's occupation` | Categorical | Feature | Lúc nhập học |
| `Admission grade` | Continuous (Float) | Feature | Lúc nhập học |
| `Displaced` | Binary | Feature | Lúc nhập học |
| `Educational special needs` | Binary | Feature | Lúc nhập học |
| `Debtor` | Binary | Feature | Lúc nhập học |
| `Tuition fees up to date` | Binary | Feature | Lúc nhập học |
| `Gender` | Binary | Feature | Lúc nhập học |
| `Scholarship holder` | Binary | Feature | Lúc nhập học |
| `Age at enrollment` | Continuous (Integer) | Feature | Lúc nhập học |
| `International` | Binary | Feature | Lúc nhập học |
| `Unemployment rate` | Continuous (Float) | Feature | Lúc nhập học |
| `Inflation rate` | Continuous (Float) | Feature | Lúc nhập học |
| `GDP` | Continuous (Float) | Feature | Lúc nhập học |
| `Target` | Categorical (3 classes) | Target | Cuối khóa học |
