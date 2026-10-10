# 📋 Kế Hoạch Chiến Lược: Xử Lý Phản Biện & Nâng Cấp Toàn Diện Paper SERF

> **Dự án:** `ReqLens` / `rbl-requirements-classification`  
> **Tác giả:** Quách Thế Dương  
> **Thời điểm cập nhật:** Tháng 10/2026  
> **Mục tiêu:** Xử lý triệt để phản biện từ hội nghị EAI FISAT 2026 và nâng cấp chất lượng nghiên cứu để hướng tới các hội nghị/tạp chí uy tín (hạng CORE B như ESEM, RE, SANER hoặc IST/JSS).

---

## 📌 BỐI CẢNH & NGUYÊN TẮC HÀNH ĐỘNG

Bài báo gốc bị từ chối (*Reject*) tại EAI FISAT 2026 không phải vì số liệu thiếu trung thực (reviewer đều ghi nhận sự minh bạch của tác giả), mà vì **4 điểm nghẽn về phương pháp luận và lập luận khoa học**:
1. **Khẳng định kết quả vượt quá độ vững chắc của dữ liệu:** Tập test nhỏ (146 mẫu), nhiều lớp thiểu số chỉ có 2–3 mẫu, chênh lệch $2.16$ Macro-F1 nhỏ hơn độ lệch chuẩn của baseline ($0.0577$).
2. **Cơ chế Selective Routing chưa từng được bóc tách (Ablation):** Tập lớp rủi ro $\mathcal{R}$ chiếm 9/12 lớp đã lấn át toàn bộ; chưa chứng minh được vai trò độc lập của confidence và margin.
3. **So sánh chưa công bằng về Data Augmentation:** SERF không augment nhưng một số baseline lại dùng augment.
4. **Độ dài bài báo vượt khung nghiêm trọng:** 23 trang so với giới hạn 12–15 trang (thừa 8 trang).

Dưới đây là 2 phương án chi tiết:
* **Phương án 1:** Khắc phục trực diện toàn bộ feedback của Reviewer trên bản thảo và số liệu Colab hiện có (Nhanh, an toàn, không tốn GPU).
* **Phương án 2:** Kế hoạch nâng cấp chất lượng nghiên cứu toàn diện (Mở rộng dataset, tái thiết kế kiến trúc, benchmark phần cứng chuẩn).

---

## 🅰️ PHƯƠNG ÁN 1: KHẮC PHỤC TRỰC DIỆN FEEDBACK CỦA REVIEWER
*(Tập trung vào bản thảo `manuscript/main.tex` và khai thác số liệu Colab sẵn có)*

### 1. Mục tiêu
Giải quyết dứt điểm các lỗi lập luận, đưa bản thảo về chuẩn mực khoa học khiêm tốn, minh bạch và cắt gọt độ dài về đúng khung 12–15 trang. Toàn bộ phương án này **chạy 100% trên CPU**, không cần huấn luyện lại trên GPU.

### 2. Các đầu việc cụ thể

#### Công việc 1.1: Điều chỉnh tuyên bố khoa học (Claim De-escalation — R1 #2, R3 #2)
* **Vị trí sửa:** `Abstract`, `Section 1 (Introduction)`, `Section 5.3 & 5.4`, `Section 7 (Conclusion)`.
* **Cách sửa:**
  * Bỏ các cụm từ khẳng định tuyệt đối như *"SERF vượt trội hơn RoBERTa 4.91 và Static Fusion 2.16 điểm Macro-F1"*.
  * Thay thế bằng: *"Trên tập dữ liệu PROMISE_EXP gồm 146 mẫu test qua 3 seed, SERF đạt kết quả mô tả (descriptive observation) là 0.7770 ± 0.0307 Macro-F1. Khoảng cách này nằm trong biên độ phương sai giữa các seed của baseline tĩnh (0.7554 ± 0.0577), phản ánh sự nhạy cảm của các lớp thiểu số có số lượng mẫu rất ít (2–5 mẫu) thay vì ưu thế vượt trội mang tính thống kê."*
  * Thừa nhận rõ ràng hạn chế kích thước tập test và khuyến nghị cần đánh giá trên quy mô lớn hơn.

#### Công việc 1.2: Minh bạch hóa logic của bộ định tuyến (Router Transparency — R1 #4, R3 #3)
* **Vị trí sửa:** `Section 3.4 (SERF Architecture)`, `Section 4.4 (Complexity & Routing Analysis)`.
* **Cách sửa:**
  * Đưa **chứng minh toán học về tính dư thừa của Margin**: Với bất kỳ phân bố xác suất hợp lệ nào, khi $\text{confidence} = p_1 \ge \tau \ge 0.70$ thì $\text{margin} = p_1 - p_2 \ge 2p_1 - 1 \ge 0.40 > \delta$ (với $\delta \in \{0.15, 0.20\}$). Do đó, điều kiện $\text{margin} < \delta$ bị bao hàm hoàn toàn bởi $\text{confidence} < \tau$. Điều này giải thích tại sao $\delta = 0.15$ và $0.20$ cho kết quả giống hệt nhau.
  * Thẳng thắn thừa nhận việc tập rủi ro $\mathcal{R}$ gồm 9/12 lớp chi phối mạnh mẽ tỷ lệ route 36.76%, định vị SERF là cơ chế kết hợp giữa *class-prior* (ưu tiên an toàn cho các lớp khó) và *confidence*.

#### Công việc 1.3: Hiệu chỉnh và làm rõ phân tích chi phí tính toán (Efficiency Clarification — R1 #5)
* **Vị trí sửa:** `Section 4.4`, `Table 5 (Active-encoder complexity)`.
* **Cách sửa:**
  * Giải trình rõ lý do vì sao trong phiên bản thực thi hiện tại, $C_{\text{hard}} = 3$ (bởi vì selector ELECTRA và nhánh fusion ELECTRA–RoBERTa ban đầu được huấn luyện độc lập để giữ nguyên tính mô đun).
  * Đối chiếu công bằng cả 3 trạng thái:
    1. **ELECTRA-only:** $C = 1.0$ encoder (rẻ nhất, hiệu năng thấp hơn).
    2. **SERF (hiện tại):** $C_{\text{SERF}} = 1 + 2r \approx 1.735$ encoders (tiết kiệm 13.24% so với Static Fusion).
    3. **Static Fusion:** $C = 2.0$ encoders (luôn tốn 2 encoder cho mọi câu).
  * Nêu rõ tiềm năng kiến trúc: Nếu tái sử dụng biểu diễn ELECTRA thì $C_{\text{hard}}$ sẽ giảm xuống $2$, đưa chi phí về $1 + r \approx 1.37$ encoders (tiết kiệm ~31.5%).

#### Công việc 1.4: Cắt gọt độ dài bài báo từ 23 trang xuống 12–15 trang (R3 #1)
* **Vị trí sửa:** Toàn bộ file `manuscript/main.tex`.
* **Cách sửa:**
  * **Mục Related Work (Sec. 2):** Rút gọn từ 3 trang xuống còn 1.5 trang, tập trung trực diện vào NFR classification và hybrid transformer.
  * **Tránh lặp narrative kết quả (Sec. 5 & Sec. 7):** Hiện tại phần phân tích kết quả đang được kể chi tiết ở Sec 5 rồi lặp lại gần như nguyên vẹn ở phần trả lời RQ1-RQ3 (Sec 7). Ta sẽ gộp câu trả lời RQ thành các đoạn đúc kết ngắn gọn, súc tích.
  * **Thu nhỏ hình ảnh & bảng biểu:** Thay vì hình 3 ma trận nhầm lẫn chiếm nguyên 1 trang (Fig. 4), ta chuyển sang 1 bảng thống kê lỗi theo cặp lớp điển hình (ví dụ: F ↔ O, SC ↔ F).
  * **Xử lý 4 Phụ lục (Appendices A–D):** Lược bỏ hoặc cô đọng các định nghĩa công thức cơ bản (Precision/Recall/F1 ai cũng biết), chuyển các bảng cấu hình chi tiết thành link dẫn tới GitHub repository mã nguồn mở.

#### Công việc 1.5: Tổng hợp bảng Routing Ablation từ log Colab có sẵn
* **Cách làm:** Khai thác các output chi tiết từ notebook `model-training/ELECTRA_RoBERTa_SERF_without_augmentation.ipynb` để xây dựng bảng bóc tách:
  * So sánh tỷ lệ route và kết quả khi: Chỉ bật Confidence vs Chỉ bật Class-rule vs Full SERF.

### 3. Đánh giá Phương án 1
* **Ưu điểm:** Khả thi 100% ngay lập tức; giải quyết đúng những gì Reviewer bắt bẻ; không tốn chi phí GPU; hoàn thiện bài báo nhanh chóng (dự kiến 2–3 ngày).
* **Nhược điểm:** Mới dừng lại ở mức "sửa lỗi bài cũ", chưa tạo ra bước đột phá về tính mới kiến trúc hay mở rộng quy mô dữ liệu để cạnh tranh các hội nghị top cao hơn.

---

## 🅱️ PHƯƠNG ÁN 2: KẾ HOẠCH NÂNG CẤP CHẤT LƯỢNG NGHIÊN CỨU TOÀN DIỆN
*(Hướng tới một công trình hoàn chỉnh, đột phá nộp CORE B: ESEM, RE, SANER, IST, JSS)*

### 1. Mục tiêu
Giải quyết tận gốc rễ những lo ngại của Reviewer 2 về tính thực tế (real-world applicability) và tính mới kiến trúc (architectural novelty), biến ReqLens thành một framework phân loại yêu cầu phần mềm chuẩn mực, hiệu quả và được kiểm chứng trên dữ liệu công nghiệp thực tế.

```
┌────────────────────────────────────────────────────────────────────────┐
│                   BỐN TRỤ CỘT NÂNG CẤP NGHIÊN CỨU (REFLENS V2)          │
├──────────────────┬──────────────────┬──────────────────┬───────────────┤
│ TRỤ CỘT 1        │ TRỤ CỘT 2        │ TRỤ CỘT 3        │ TRỤ CỘT 4     │
│ DATASET & EXPERT │ KIẾN TRÚC MÔ HÌNH│ PROFILE PHẦN CỨNG│ PHÂN TÍCH LỖI │
├──────────────────┼──────────────────┼──────────────────┼───────────────┤
│ • Cross-Project  │ • Feature Reuse  │ • Latency p50/p95│ • Mixed FR/NFR│
│   (disjoint ID)  │   (C_hard = 2)   │ • Peak GPU VRAM  │   Taxonomy    │
│ • Agile Stories  │ • Calibrated     │ • Throughput thực│ • Phân tích   │
│   (Jira/GitHub)  │   Entropy Router │ • Đo FLOPs       │   ngôn ngữ học│
│ • Multi-Dataset  │ • Lightweight    │                  │   phần mềm    │
│   (DCAI24, EARS) │   Baselines (SVM)│                  │               │
└──────────────────┴──────────────────┴──────────────────┴───────────────┘
```

### 2. Bốn trụ cột nâng cấp chiến lược

#### Trụ cột 2.1: Mở rộng dữ liệu & Kiểm chứng độ bền vững (Data Robustness)
1. **Kiểm chứng Cross-Project (Project-Disjoint Split):**
   * *Vấn đề cũ:* Chia ngẫu nhiên 70/15/15 có thể khiến các câu trong cùng một dự án phần mềm rơi vào cả tập train lẫn test, gây rò rỉ từ vựng chuyên ngành.
   * *Giải pháp mới:* Dùng cột `ProjectID` trong `promise_exp.csv` để chia tập theo dự án (ví dụ: lấy 3 dự án độc lập làm tập test). Chứng minh mô hình có thể tổng quát hóa sang một hệ thống phần mềm mới toanh.
2. **Tích hợp tập dữ liệu thực tế / Agile User Stories:**
   * *Vấn đề cũ:* Reviewer 2 chỉ trích PROMISE là tập dữ liệu học thuật cũ, quá sạch và không đại diện cho thực tế công nghiệp Agile.
   * *Giải pháp mới:* Bổ sung đánh giá trên tập dữ liệu `DCAI24` (đã có trong `data/raw/dcai24_src_dataset.xlsx`) hoặc bộ dữ liệu Agile User Stories từ Jira / GitHub Issues. Đo độ sụt giảm hiệu năng khi gặp câu văn lộn xộn, không chuẩn ngữ pháp.

#### Trụ cột 2.2: Tái thiết kế kiến trúc huấn luyện (Architecture Redesign)
1. **Kiến trúc Cascade chia sẻ biểu diễn (Representation Reuse):**
   * *Thiết kế mới:* Huấn luyện mô hình theo dạng End-to-End Multi-Exit:
     $$\text{Câu yêu cầu} \longrightarrow \mathbf{ELECTRA} \longrightarrow \mathbf{h}_{\text{electra}} \longrightarrow \text{Classifier}_1 \text{ (Selector)}$$
   * Nếu dự đoán tự tin $\to$ Xuất nhãn ngay (Chi phí = 1 encoder).
   * Nếu không tự tin $\to$ **Tái sử dụng nguyên vẹn** $\mathbf{h}_{\text{electra}}$, chỉ kích hoạt thêm $\mathbf{RoBERTa} \to \mathbf{h}_{\text{roberta}}$, rồi đưa cặp $[\mathbf{h}_{\text{electra}}, \mathbf{h}_{\text{roberta}}]$ vào Fusion Head (Chi phí chỉ thêm 1 encoder).
   * **Kết quả:** $C_{\text{hard}} = 2$ encoders. Chi phí trung bình giảm xuống $C_{\text{SERF}} = 1 + r \approx 1.37$ encoders (tiết kiệm hơn 31% chi phí so với Static Fusion).
2. **Bộ định tuyến thông minh dựa trên Entropy / Calibration (Calibrated Router):**
   * *Bỏ hoàn toàn danh sách cứng 9 lớp $\mathcal{R}$* (xóa bỏ chỉ trích "class-switch").
   * Áp dụng **Temperature Scaling** để hiệu chuẩn độ tin cậy, kết hợp tính **Shannon Entropy**:
     $$\mathcal{H}(p) = -\sum_{k=1}^{K} p_k \log p_k$$
   * Câu nào có Entropy vượt ngưỡng (mô hình phân vân giữa các lớp) sẽ tự động kích hoạt Fusion. Đây là đóng góp thuật toán thực sự.
3. **Bổ sung Lightweight Baselines đầy đủ:**
   * Bổ sung **TF-IDF + Linear SVM** và **DistilBERT** làm mốc so sánh cơ sở (baseline chuẩn mực để so sánh đánh đổi hiệu năng - chi phí).

#### Trụ cột 2.3: Đo đạc tài nguyên thực tế chuẩn chỉnh (Hardware Profiling)
* Thay thế công thức lý thuyết $1+2r$ bằng các phép đo phần cứng thực tế trên Colab (T4 / A100):
  * **Độ trễ suy luận (Inference Latency):** Báo cáo trung vị $p50$ và phân vị $95$ ($p95$) tính bằng milli-giây trên từng câu.
  * **Bộ nhớ GPU đỉnh (Peak VRAM):** Sử dụng `torch.cuda.max_memory_allocated()`.
  * **Thông lượng (Throughput):** Số câu xử lý/giây ở các kích thước batch khác nhau (batch size 1, 8, 32).
  * **FLOPs:** Tính toán tổng số phép tính dấu phẩy động cho từng đường rẽ nhánh.

#### Trụ cột 2.4: Phân tích lỗi định tính chuyên sâu (Qualitative Error Analysis)
* Giải quyết triệt để phản biện Major #4 của Reviewer 2:
  * Xây dựng danh mục phân loại lỗi cho các câu **yêu cầu hỗn hợp (Mixed FR/NFR)** (ví dụ: *"The system shall authenticate user [FR] within 1 second [PE]"*).
  * Giải thích cơ chế ngôn ngữ học: Tại sao ELECTRA nhầm lẫn ở các động từ mơ hồ (*support, handle, provide*), và RoBERTa bổ trợ ngữ cảnh như thế nào để sửa sai.

### 3. Đánh giá Phương án 2
* **Ưu điểm:** Tạo ra một bài báo nghiên cứu thực nghiệm mẫu mực (empirical SE research); đóng góp mới rõ ràng; đủ sức cạnh tranh tại các hội nghị CORE B danh giá hoặc tạp chí Q1/Q2; giải quyết 100% tất cả các câu hỏi khó nhất của Reviewer 2.
* **Nhược điểm:** Cần thời gian huấn luyện lại trên GPU Colab (dự kiến 1–2 tuần chạy thực nghiệm và phân tích).

---

## 🚀 LỘ TRÌNH KẾT HỢP TỐI ƯU (HYBRID ROADMAP ĐỀ XUẤT)

Cách tiếp cận khôn ngoan nhất là **triển khai theo 2 pha tuần tự (Pha 1 dọn đường cho Pha 2)**:

| Giai đoạn | Nội dung thực hiện | Sản phẩm đầu ra | Thời gian |
| :--- | :--- | :--- | :---: |
| **Pha 1 (Bảo vệ & Sửa bài)** | • Thực hiện toàn bộ **Phương án 1** trên `main.tex`.<br>• Sửa Abstract, đưa chứng minh toán học margin redundancy vào bài, hạ bớt các tuyên bố thái quá, cắt gọt bài về 14–15 trang.<br>• Viết file giải trình phản biện (Reviewer Response Matrix). | Bản thảo LaTeX sạch sẽ, chuẩn 15 trang, sẵn sàng nếu cần resubmit nhanh. | 2–3 ngày |
| **Pha 2 (Nâng cấp đột phá)** | • Thực hiện **Phương án 2** trên Colab.<br>• Code kiến trúc mới (Feature Reuse $C_{\text{hard}}=2$, Calibrated Router).<br>• Chạy benchmark đo latency/VRAM và thêm baseline SVM/DistilBERT.<br>• Thử nghiệm trên tập dữ liệu mở rộng (`DCAI24` / Agile User Stories). | Bộ số liệu thực nghiệm V2 vượt trội, nâng tầm bài báo thành công trình mạnh mẽ nộp venue lớn. | 1–2 tuần |

---

*File này được lưu trữ tại [ke-hoach-cai-tien-va-sua-paper.md](file:///d:/reseach/ReqLens/ke-hoach-cai-tien-va-sua-paper.md) để bạn tiện đọc và theo dõi.*
