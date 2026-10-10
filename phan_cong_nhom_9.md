# NHẬT KÝ THỰC HIỆN 06 TUẦN & BẢNG PHÂN CÔNG ĐÓNG GÓP ĐỒ ÁN

> **HỌC PHẦN:** HỌC MÁY CƠ BẢN — NĂM HỌC 2026 – 2027  
> **TRƯỜNG:** ĐẠI HỌC SƯ PHẠM KỸ THUẬT HƯNG YÊN (HYUTE)  
> **LỚP:** 12523W.1  
> **GIẢNG VIÊN HƯỚNG DẪN:** PGS.TS. NGUYỄN VĂN HẬU  
> **ĐỀ TÀI:** PROJECT 09 — CẢNH BÁO SỚM NGUY CƠ BỎ HỌC TỪ DỮ LIỆU ĐẦU VÀO  
> **CHỦ ĐỀ:** BÀI 3 — MA TRẬN NHẦM LẪN, CÁC CHỈ SỐ VÀ KHẢ NĂNG TỔNG QUÁT HÓA  

---

## 1. Thông tin nhóm sinh viên thực hiện

| STT | Họ và tên | Mã sinh viên | Lớp | Vai trò | Tỷ lệ đóng góp |
| :-: | :--- | :-: | :-: | :---: | :-: |
| 1 | **Nguyễn Duy Huy** | **10123158** | 12523W.1 | Nhóm trưởng | **50%** |
| 2 | **Nguyễn Minh Đoàn** | **10123100** | 12523W.1 | Thành viên | **50%** |

---

## 2. Bảng ma trận phân công nhiệm vụ

| Hạng mục công việc | Người phụ trách chính | Người phối hợp / Review | Mức độ hoàn thành |
| :--- | :---: | :---: | :-: |
| **1. Khảo sát dữ liệu & kiểm tra Data Leakage** | Nguyễn Duy Huy | Nguyễn Minh Đoàn | 100% |
| **2. Phân tích khám phá dữ liệu (EDA)** | Nguyễn Minh Đoàn | Nguyễn Duy Huy | 100% |
| **3. Tiền xử lý dữ liệu & chia Train/Validation/Test** | Nguyễn Duy Huy | Nguyễn Minh Đoàn | 100% |
| **4. Xây dựng Baseline & Logistic Regression đa lớp** | Nguyễn Minh Đoàn | Nguyễn Duy Huy | 100% |
| **5. Thực hiện 4 thí nghiệm bắt buộc** | Cả hai cùng thực hiện | Cả hai cùng thực hiện | 100% |
| **6. Đánh giá Confusion Matrix, Macro-F1, Recall Dropout** | Nguyễn Duy Huy | Nguyễn Minh Đoàn | 100% |
| **7. Thiết kế REST API** | Nguyễn Duy Huy | Nguyễn Minh Đoàn | 100% |
| **8. Thiết kế giao diện Web & Dashboard** | Nguyễn Minh Đoàn | Nguyễn Duy Huy | 100% |
| **9. Soạn báo cáo & Model Card** | Nguyễn Minh Đoàn | Nguyễn Duy Huy | 100% |
| **10. Thiết kế Slide & diễn tập bảo vệ** | Cả hai cùng thực hiện | Cả hai cùng thực hiện | 100% |

---

## 3. Nhật ký chi tiết tiến độ 06 tuần

### TUẦN 1: Khởi động đề tài & nghiên cứu dữ liệu

**Thời gian:** 18/08/2026 – 24/08/2026  

**Mục tiêu:** Đọc yêu cầu Project 09, nghiên cứu bộ dữ liệu UCI, xác định bài toán phân loại 3 lớp và thời điểm dự đoán là lúc nhập học.

| Thành viên | Nhiệm vụ cụ thể | Giờ ước tính | Sản phẩm đầu ra |
| :--- | :--- | :-: | :--- |
| **Nguyễn Duy Huy** | Đọc đặc tả Project 09; xác định câu hỏi nghiên cứu, đầu vào, đầu ra; lập Project Brief. | 14h | Project Brief |
| **Nguyễn Minh Đoàn** | Nghiên cứu bộ dữ liệu; lập Data Dictionary; cấu hình môi trường Python, Git và `requirements.txt`. | 13h | Data Dictionary, Repo Git |

**Vấn đề & giải quyết:**  
Phát hiện các biến kết quả học kỳ 1 và 2 có thể gây Data Leakage. Nhóm thống nhất chỉ dùng thông tin có tại thời điểm nhập học.

---

### TUẦN 2: Tiền xử lý dữ liệu, EDA & chia tập

**Thời gian:** 25/08/2026 – 31/08/2026  

**Mục tiêu:** Làm sạch dữ liệu, phân tích phân bố lớp và chia Train/Validation/Test.

| Thành viên | Nhiệm vụ cụ thể | Giờ ước tính | Sản phẩm đầu ra |
| :--- | :--- | :-: | :--- |
| **Nguyễn Duy Huy** | Viết `src/data.py`; loại các biến gây leakage; chia dữ liệu Train/Validation/Test có stratify. | 16h | `src/data.py`, dữ liệu đã chia |
| **Nguyễn Minh Đoàn** | Thực hiện EDA; kiểm tra missing, trùng lặp, phân bố `Dropout`, `Enrolled`, `Graduate`. | 15h | Notebook EDA, biểu đồ |

**Vấn đề & giải quyết:**  
Dữ liệu mất cân bằng giữa ba lớp nên nhóm sử dụng Stratified Split để giữ tỷ lệ lớp giữa các tập.

---

### TUẦN 3: Xây dựng Baseline & mô hình chính

**Thời gian:** 01/09/2026 – 07/09/2026  

**Mục tiêu:** Xây dựng Pipeline tiền xử lý, Baseline và Multinomial Logistic Regression.

| Thành viên | Nhiệm vụ cụ thể | Giờ ước tính | Sản phẩm đầu ra |
| :--- | :--- | :-: | :--- |
| **Nguyễn Duy Huy** | Xây dựng Pipeline preprocessing, encoding, scaling và Logistic Regression đa lớp. | 18h | `src/features.py`, `src/train.py` |
| **Nguyễn Minh Đoàn** | Xây dựng baseline lớp phổ biến và module đánh giá Accuracy, Precision, Recall, F1, Macro-F1. | 17h | Baseline, module metrics |

**Vấn đề & giải quyết:**  
Accuracy có thể cao do mất cân bằng lớp, vì vậy nhóm sử dụng thêm Macro-F1 và Recall lớp Dropout.

---

### TUẦN 4: Chạy 4 thí nghiệm & phân tích lỗi

**Thời gian:** 08/09/2026 – 14/09/2026  

**Mục tiêu:** Hoàn thành 4 thí nghiệm bắt buộc của Project 09.

| Thành viên | Nhiệm vụ cụ thể | Giờ ước tính | Sản phẩm đầu ra |
| :--- | :--- | :-: | :--- |
| **Nguyễn Duy Huy** | Thí nghiệm Accuracy vs Macro-F1; phân tích threshold lớp Dropout. | 18h | Bảng metric, biểu đồ threshold |
| **Nguyễn Minh Đoàn** | Thí nghiệm Data Leakage; Confusion Matrix; audit sai số theo giới, tuổi, quốc tịch. | 18h | Confusion Matrix, báo cáo sai số |

**Các thí nghiệm:**
1. So sánh Accuracy và Macro-F1.  
2. So sánh mô hình có biến học kỳ với mô hình đúng thời điểm nhập học.  
3. Phân tích threshold/ưu tiên lớp Dropout.  
4. Audit sai số theo giới, tuổi và quốc tịch.

---

### TUẦN 5: Xây dựng API & giao diện Web

**Thời gian:** 15/09/2026 – 21/09/2026  

**Mục tiêu:** Tích hợp mô hình vào API và giao diện Web.

| Thành viên | Nhiệm vụ cụ thể | Giờ ước tính | Sản phẩm đầu ra |
| :--- | :--- | :-: | :--- |
| **Nguyễn Duy Huy** | Xây dựng API `/api/support-priority`; validation dữ liệu; nạp model đã huấn luyện. | 19h | `app/main.py`, API |
| **Nguyễn Minh Đoàn** | Thiết kế Web 3 màn hình: giới thiệu, dự đoán, dashboard/model card. | 20h | HTML/CSS/JS |

**Vấn đề & giải quyết:**  
Tách hoàn toàn quá trình Training khỏi Serving; ứng dụng Web chỉ nạp model đã lưu, không huấn luyện lại khi có request.

---

### TUẦN 6: Hoàn thiện báo cáo & thuyết trình

**Thời gian:** 22/09/2026 – 28/09/2026  

**Mục tiêu:** Hoàn thiện báo cáo, slide, kiểm thử và diễn tập bảo vệ.

| Thành viên | Nhiệm vụ cụ thể | Giờ ước tính | Sản phẩm đầu ra |
| :--- | :--- | :-: | :--- |
| **Nguyễn Duy Huy** | Viết phần Pipeline, Logistic Regression, leakage và kết quả thí nghiệm; hoàn thiện README. | 16h | Báo cáo, README |
| **Nguyễn Minh Đoàn** | Viết phần EDA, Confusion Matrix, phân tích lỗi, Web/API; thiết kế slide. | 16h | Báo cáo, Slide |
| **Cả hai** | Rà soát báo cáo, kiểm thử hệ thống, diễn tập demo và vấn đáp. | 8h | Bản nộp hoàn chỉnh |

---

## 4. Bảng kiểm tra cổng chất lượng trước khi nộp

| Cổng kiểm tra | Tiêu chuẩn | Tình trạng |
| :--- | :--- | :---: |
| **1. Dữ liệu** | Có nguồn, Data Dictionary, script xử lý dữ liệu. | **ĐẠT** |
| **2. Chống rò rỉ** | Loại biến học kỳ; preprocessing chỉ fit trên Train. | **ĐẠT** |
| **3. Đánh giá** | Có Baseline, Macro-F1, Recall Dropout, Confusion Matrix và 4 thí nghiệm. | **ĐẠT** |
| **4. Web/API** | Web và API chạy được; có validation đầu vào. | **ĐẠT** |
| **5. Tái lập** | Có `requirements.txt`, README và model đã lưu. | **ĐẠT** |
| **6. Đóng góp nhóm** | Hai thành viên đóng góp cân bằng 50% – 50%. | **ĐẠT** |

---

*Hưng Yên, ngày 29 tháng 09 năm 2026*

**XÁC NHẬN CỦA CÁC THÀNH VIÊN NHÓM**

| Nhóm trưởng | Thành viên |
| :---: | :---: |
| *(Ký và ghi rõ họ tên)* | *(Ký và ghi rõ họ tên)* |
| <br><br>**Nguyễn Duy Huy** | <br><br>**Nguyễn Minh Đoàn** |
| **10123158** | **10123100** |