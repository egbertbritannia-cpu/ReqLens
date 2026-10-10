# 📐 Quy Trình Thực Nghiệm Chuẩn Hóa & Bộ Quy Tắc Kiểm Chứng Công Bằng (Experimental Protocol V2)

> **Dự án:** `ReqLens` / `rbl-requirements-classification`  
> **Người phụ trách / Quyết định:** Quách Thế Dương  
> **Mã công việc:** `P05 — Freeze matched experimental protocol`  
> **Trạng thái:** ⚠️ **DỰ THẢO — CHƯA FREEZE (DRAFT — NOT YET FROZEN)**  
> **Mục tiêu:** Thiết lập bộ quy tắc kiểm chứng công bằng, chặt chẽ và nhất quán trước khi thực hiện các đợt chạy GPU/Colab tiếp theo; loại bỏ triệt để các lỗi so sánh khập khiễng (*confounding variables*), rò rỉ dữ liệu (*data leakage*), và biến động hạt giống mà Reviewer FISAT 2026 đã chỉ trích.

---

## 🎯 TRIẾT LÝ CỐT LÕI

> **"Thiết kế chặt chẽ trước — Train một lần duy nhất — Số liệu vững chắc — Không phải chạy lại."**

Mọi kết quả thực nghiệm đưa vào bài báo nghiên cứu phải tuân thủ nghiêm ngặt nguyên tắc so sánh công bằng: cùng tập dữ liệu, cùng cách chia, cùng bộ hạt giống, cùng điều kiện tăng cường dữ liệu, cùng phương pháp trích xuất đặc trưng và cùng thước đo đánh giá.

---

## 🛡️ BỘ 7 NGUYÊN TẮC KIỂM CHỨNG BẤT BIẾN

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│              BỘ 7 NGUYÊN TẮC KIỂM CHỨNG CÔNG BẰNG (CHƯA FREEZE)                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│ 1. Cố định Split (Zero Leakage)    : Chia tập trước ➔ Tiền xử lý sau              │
│ 2. Cố định Seeds                   : Mọi model đều chạy đúng 3 seed [42,123,2024] │
│ 3. Cố định điều kiện Augmentation  : Bảng so sánh chính 100% = KHÔNG AUG (No-Aug)│
│ 4. Cố định Preprocessing & Input   : Cùng max_len=128, cùng cách lấy token [CLS]  │
│ 5. Cấm nhìn tập Test (No Peeking)  : Mọi checkpoint & threshold chỉ chọn trên Val │
│ 6. Cố định Metric & Thống kê       : Cùng Macro-F1 (zero_division=0), Bootstrap CI│
│ 7. Cố định môi trường đo phần cứng : Cùng GPU, cùng batch_size=1, có warm-up      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

### 1. Nguyên tắc 1: Cố định phân tách dữ liệu & Chống rò rỉ tuyệt đối (Zero Leakage)
* **Tập dữ liệu chuẩn:** `data/exp/promise_exp.csv` (969 mẫu, 12 lớp).
* **Phân chia Stratified 70/15/15 cố định:**
  * Cố định `SPLIT_SEED = 42` qua hai lượt chia tầng bằng `scikit-learn`:
    * Lượt 1: `test_size = 0.30` $\to$ Tách ra 70% Train (678 mẫu) và 30% tạm thời (291 mẫu).
    * Lượt 2: `test_size = 0.50` trên 30% tạm thời $\to$ Tách thành 15% Validation (145 mẫu) và 15% Test (146 mẫu).
  * Phân bố từng lớp phải khớp 100% với chữ ký tái lập trong `experiments/routing/split_provenance.py`.
* **Phân chia mở rộng Cross-Project (Project-Disjoint Split):**
  * Đối với thí nghiệm kiểm chứng khả năng tổng quát hóa liên dự án, tập Test phải bao gồm toàn bộ các yêu cầu từ các `ProjectID` độc lập; tuyệt đối không để cùng một `ProjectID` xuất hiện ở cả tập Train và tập Test.
* **Nguyên tắc chống rò rỉ:**
  * Phân chia tập dữ liệu **phải diễn ra trước mọi thao tác tiền xử lý hoặc tăng cường dữ liệu**.
  * Việc tính trọng số lớp (`class_weights`), từ điển TF-IDF, hay thống kê từ vựng **chỉ được fit trên tập Train**. Cấm sử dụng thông tin của Validation và Test cho bước tiền xử lý.

---

### 2. Nguyên tắc 2: Cố định danh sách hạt giống huấn luyện (Identical Training Seeds)
* **Bộ hạt giống chuẩn hóa:**
  $$\text{TRAIN\_SEEDS} = [42, 123, 2024]$$
* **Quy tắc bắt buộc:**
  * Mọi mô hình nằm trong bảng so sánh chính — từ mô hình nhẹ (TF-IDF + Linear SVM, DistilBERT) đến các mô hình Transformer đơn (BERT, RoBERTa, ELECTRA, ModernBERT), mô hình Static Fusion và mô hình đề xuất SERF V2 — **đều phải chạy đủ trên cả 3 hạt giống này**.
  * Chấm dứt tình trạng mô hình chạy 1 seed, mô hình chạy 3 seed dẫn tới phê bình của Reviewer 1 #3.
* **Hình thức báo cáo:** Tất cả các chỉ số trên tập Test phải được báo cáo dưới dạng:
  $$\text{Mean} \pm \text{Standard Deviation}$$

---

### 3. Nguyên tắc 3: Cố định điều kiện Augmentation (Fair Augmentation Protocol)
* **Bảng so sánh chính (Main Results Table):**
  * **100% các mô hình so sánh trong bảng kết quả chính bắt buộc phải ở điều kiện KHÔNG AUGMENTATION (No-Aug).**
  * Điều này đảm bảo tính khách quan và so sánh ngang hàng tuyệt đối giữa các kiến trúc.
* **Bảng nghiên cứu thành phần (Augmentation Ablation Table):**
  * Nếu đánh giá tác động của kỹ thuật Augmentation, thí nghiệm phải được tách thành một bảng bóc tách riêng.
  * Chỉ áp dụng Augmentation lên tập Train, tuyệt đối không augment trên tập Validation và Test.

---

### 4. Nguyên tắc 4: Cố định tiền xử lý & Siêu tham số đầu vào (Input & Feature Alignment)
* **Chiều dài câu tối đa:** `max_length = 128` (đảm bảo bao trọn 100% chiều dài các câu yêu cầu phần mềm trong tập dữ liệu mà không bị cắt cụt).
* **Quy tắc làm sạch văn bản:**
  * Chỉ chuẩn hóa khoảng trắng thừa (`\s+` $\to$ single space).
  * Giữ nguyên chữ hoa/thường (nếu dùng cased/discriminator), giữ nguyên số liệu kỹ thuật, dấu câu đặc biệt, và các modal verbs (*shall, must, should, will*) vì đây là đặc trưng ngữ nghĩa cốt lõi của yêu cầu phần mềm.
* **Cơ chế trích xuất biểu diễn (Feature Extraction):**
  * Sử dụng biểu diễn của token đầu tiên (`[CLS]` hoặc `<s>`) từ lớp ẩn cuối cùng (*last hidden state*).
  * Không dùng pooler output để đảm bảo sự tương thích và công bằng giữa các kiến trúc Transformer khác nhau.

---

### 5. Nguyên tắc 5: Tuyệt đối không nhìn trước tập Test (Strict Zero Test-Peeking)
* **Quy tắc ngăn chặn rò rỉ thông tin kiểm thử:**
  * Quyết định lưu checkpoint tối ưu trong quá trình train (Early Stopping / Best Model) **chỉ được dựa trên Validation Macro-F1**.
  * Việc dò tìm ngưỡng định tuyến $\tau$ (confidence threshold) hoặc entropy threshold của SERF **chỉ được tối ưu hóa trên tập Validation**.
  * **Tập Test chỉ được mở ra một lần duy nhất** khi toàn bộ mô hình và ngưỡng định tuyến đã được đóng băng hoàn toàn.

---

### 6. Nguyên tắc 6: Thống nhất chỉ số & Kiểm định thống kê (Metric & Testing Contract)
* **Thư viện chuẩn:** Sử dụng nhất quán `scikit-learn` với tham số `zero_division = 0`.
* **Hệ thống chỉ số báo cáo:**
  * **Chỉ số xếp hạng chính (Primary Metric):** `Macro-F1` (trung bình cộng không trọng số của toàn bộ 12 lớp, bảo vệ quyền lợi của các lớp thiểu số).
  * **Chỉ số phụ trợ:** `Accuracy`, `Weighted-F1`, `Macro-Precision`, `Macro-Recall`, `Per-class F1` và `Route-to-Fusion Rate (r)`.
* **Kiểm định ý nghĩa thống kê:**
  * Bổ sung **Bootstrap 95% Confidence Interval** (1,000 lần lấy mẫu lại trên tập Test).
  * Thực hiện kiểm định cặp **Paired Wilcoxon Signed-Rank Test** để báo cáo giá trị $p$-value, xác định rõ khoảng cách giữa các mô hình có ý nghĩa thống kê hay nằm trong biên độ ngẫu nhiên.

---

### 7. Nguyên tắc 7: Chuẩn hóa môi trường đo đạc tài nguyên (Empirical Compute Benchmark)
* **Môi trường phần cứng:**
  * Toàn bộ các phép đo hiệu năng suy luận phải được thực hiện trên cùng một loại phần cứng (khuyến nghị GPU Tesla T4 trên Google Colab).
* **Quy trình đo đạc chuẩn SE:**
  * Chạy ở chế độ `torch.no_grad()` và `model.eval()`.
  * Thực hiện **10 câu warm-up** trước khi bấm giờ để loại bỏ chi phí khởi động CUDA ban đầu.
  * Đo **Độ trễ suy luận (Latency):** Đo ở `batch_size = 1` để mô phỏng tương tác thực tế của người dùng; báo cáo trung vị ($p50$) và phân vị 95 ($p95$) tính bằng milli-giây (ms).
  * Đo **Bộ nhớ GPU đỉnh (Peak VRAM):** Sử dụng `torch.cuda.max_memory_allocated()`.
  * Đo **Thông lượng (Throughput):** Số câu xử lý được trên mỗi giây ở `batch_size = 16` và `32`.

---

## 📝 ĐIỀU KIỆN ĐỂ CHUYỂN TRẠNG THÁI SANG "FROZEN"

Tài liệu này sẽ chuyển từ trạng thái **`DRAFT — CHƯA FREEZE`** sang **`FROZEN`** khi:
1. Tác giả Quách Thế Dương rà soát và xác nhận đồng ý toàn bộ 7 nguyên tắc trên.
2. Không còn bất kỳ điểm mâu thuẫn nào giữa quy trình thực nghiệm và các yêu cầu sửa đổi của phản biện FISAT 2026.
3. Kịch bản code huấn luyện (training script / Colab notebook) được cấu hình khóa chặt theo đúng các tham số đã nêu ở trên.
