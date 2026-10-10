# 🎓 Cẩm Nang Nghiên Cứu Toàn Diện

## "How Accurately Can AI Classify Software Requirements into Functional and Non-Functional Categories?"

> **Dành cho người mới bắt đầu** | Biên soạn theo quan điểm chuyên gia SE & AI Research

---

## 📌 PHẦN 0: HIỂU RÕ CHỦ ĐỀ NGHIÊN CỨU TRƯỚC KHI BẮT ĐẦU

### 0.1 Hai Loại Yêu Cầu Phần Mềm

| Loại | Định nghĩa | Câu hỏi chủ đạo | Ví dụ |
|------|------------|-----------------|-------|
| **Functional Requirements (FR)** | Mô tả **HỆ THỐNG LÀM GÌ** | *"What?"* | "The system shall allow users to register an account via email." |
| **Non-Functional Requirements (NFR)** | Mô tả **HỆ THỐNG LÀM TỐT NHƯ THẾ NÀO** | *"How well?"* | "The system shall respond to all queries within 2 seconds." |

### 0.2 NFR Subcategories (Phân Loại Con của NFR – theo chuẩn ISO 25010)

| Code | Loại NFR | Ví dụ câu điển hình |
|------|----------|---------------------|
| PE | **Performance** | "The system shall handle 1,000 concurrent users." |
| SE | **Security** | "All passwords shall be encrypted using AES-256." |
| US | **Usability** | "The interface shall comply with WCAG 2.1 standards." |
| A | **Availability** | "The system shall maintain 99.9% uptime per month." |
| SC | **Scalability** | "The system shall scale to 10x current load within 1 hour." |
| MN | **Maintainability** | "All source code shall follow PEP 8 coding standards." |
| FT | **Fault Tolerance** | "The system shall recover automatically within 30 seconds of failure." |
| L | **Legal** | "The system shall comply with GDPR regulations." |
| LF | **Look & Feel** | "The interface shall use the company's blue color scheme." |
| O | **Operational** | "The system shall run on Windows 10 and Ubuntu 22.04." |

> 💡 **Tại sao bài toán này quan trọng?**
> Trong thực tế, các yêu cầu phần mềm thường được viết lẫn lộn, chưa được gán nhãn. Phân loại thủ công tốn kém và dễ sai nhất quán. AI phân loại đúng giúp: lập kế hoạch sprint, phân công nhóm, xây dựng test cases, và quản lý dự án hiệu quả hơn.

---

## 📚 PHẦN 1: BỐI CẢNH VÀ THÁCH THỨC KHI DÙNG AI

### 1.1 Thách Thức Kỹ Thuật Chính

**A. Vấn đề dữ liệu (Data challenges):**
- **Imbalanced dataset**: NFR thường chỉ chiếm 20–35% trong các dự án thực tế
- **Thiếu đa dạng ngôn ngữ**: Hầu hết dataset hiện có đều bằng tiếng Anh
- **Inter-annotator disagreement**: Chuyên gia khác nhau đôi khi gán nhãn khác nhau cho cùng một câu
- **Domain-specific vocabulary**: Yêu cầu hệ thống y tế rất khác với thương mại điện tử

**B. Vấn đề ngôn ngữ (Linguistic challenges):**
- **Ranh giới mơ hồ**: *"The login page shall load quickly"* → FR hay NFR (Performance)?
- **Câu hỗn hợp**: Một câu có thể chứa cả yếu tố FR lẫn NFR
- **Ambiguous verbs**: "support", "handle", "provide" → khó xác định là chức năng hay phi chức năng

**C. Vấn đề mô hình (Model challenges):**
- Classical ML cần feature engineering thủ công, thiếu hiểu ngữ nghĩa sâu
- LLMs tốn chi phí API, có thể hallucinate, nhạy cảm với cách đặt prompt
- Fine-tuned models có thể overfitting trên dataset nhỏ như PROMISE (625 rows)

---

### 1.2 Research Gaps Hiện Tại – Khoảng Trống Cần Nghiên Cứu

Đây là những điểm **còn thiếu hoặc chưa đủ** trong cộng đồng nghiên cứu (tính đến 2024–2025):

1. **LLM chưa được đánh giá toàn diện**: GPT-4o, Llama 3 (8B/70B), Claude 3, Gemini Pro chưa được so sánh trực tiếp và có hệ thống trên cùng một benchmark
2. **Fine-grained NFR subcategory**: Nhiều nghiên cứu chỉ dừng ở FR/NFR nhị phân, chưa phân loại sâu vào 10+ subcategories của NFR
3. **Cross-domain generalization**: Mô hình train trên domain ngân hàng thường kém khi áp dụng vào domain y tế hoặc IoT
4. **Low-resource & multilingual settings**: Gần như không có nghiên cứu tiếng Việt, tiếng Ả Rập, tiếng Trung cho bài toán này
5. **Agile real-world requirements**: Dataset hiện tại chủ yếu từ tài liệu học thuật, ít phản ánh thực tế User Stories trên Jira/GitHub
6. **Explainability (XAI)**: Tại sao mô hình phân loại một câu là NFR? Rất ít nghiên cứu giải thích được
7. **Prompt optimization for LLMs**: Chain-of-Thought, few-shot ordering, prompt templates cho bài toán này chưa được tối ưu hóa có hệ thống

---

### 1.3 Có Nên Đọc Papers Và Review Research Gaps Không?

> **→ CÓ, bắt buộc phải làm. Đây là nền tảng của mọi nghiên cứu nghiêm túc.**

**Chiến lược đọc papers cho người mới (2–3 tuần đầu):**

**Bước 1 – Đọc Survey Papers trước (bức tranh tổng quan):**
- Tìm keyword: `"non-functional requirements" classification "systematic review"`
- Tìm keyword: `"requirements engineering" NLP survey 2022 OR 2023 OR 2024`
- Đọc Abstract + Introduction + Conclusion trước → quyết định có đọc kỹ không

**Bước 2 – Đọc Seminal Papers (paper quan trọng gốc):**
- Cleland-Huang et al. (2006/2007) – NFR classification gốc
- Lu & Liang (2017) – Machine learning cho NFR
- Các papers dùng BERT (2019+) và GPT (2023+)

**Bước 3 – Đọc Papers Gần Nhất (2022–2025):**
- Filter trên Google Scholar: "Last 2 years" + cite count > 20
- Tìm trên arXiv: `cs.SE requirements classification LLM`

**Template ghi chú cho mỗi paper (dùng Notion/Obsidian):**

```
📄 Paper: [Tiêu đề bài báo]
📅 Năm xuất bản: 20xx
🏛️ Venue: IEEE / ACM / EMSE / IST / JSS
📊 Dataset: [Tên, số lượng mẫu, nguồn]
🤖 Model: [SVM / BERT / GPT-4 / ...]
📈 Best F1-Score: xx%
🎯 Research Gap họ đề cập: [...]
💡 Idea tôi có thể mở rộng: [...]
```

**Cách xác định Research Gap từ papers:**
- Đọc kỹ phần **"Limitations"** và **"Future Work"** → đây là gợi ý trực tiếp
- Tạo **comparison table** sau khi đọc 10 papers: Dataset × Model × F1 → gap lộ ra ngay
- Hỏi: *"Paper này chưa thử mô hình/dataset/ngôn ngữ nào?"*

---

## 🔬 PHẦN 2: PHƯƠNG PHÁP NGHIÊN CỨU VÀ CÔNG CỤ ĐỀ XUẤT

### 2.1 Cách Đánh Giá Độ Chính Xác (Evaluation Metrics)

#### A. Confusion Matrix – Nền Tảng Của Mọi Thứ

```
                    Predicted: FR      Predicted: NFR
Actual: FR    |  True Positive (TP)  |  False Negative (FN)  |
Actual: NFR   |  False Positive (FP) |  True Negative (TN)   |
```

#### B. Các Công Thức Cốt Lõi

| Metric | Công thức | Khi nào dùng? |
|--------|-----------|---------------|
| **Accuracy** | (TP+TN) / (TP+TN+FP+FN) | Dataset cân bằng (ít dùng cho bài toán này) |
| **Precision** | TP / (TP + FP) | Khi FP tốn kém (tránh báo nhầm) |
| **Recall** | TP / (TP + FN) | Khi FN tốn kém (không bỏ sót NFR) |
| **F1-Score** | 2 × (P × R) / (P + R) | **Dùng nhiều nhất** – cân bằng P và R |
| **AUC-ROC** | Area under ROC curve | So sánh mô hình độc lập với ngưỡng |
| **MCC** | Tính theo cả 4 ô | Dataset imbalanced nặng |

**Ví dụ trực quan:**
> Mô hình xử lý 100 câu. Dự đoán 30 câu là NFR, trong đó đúng 25. Thực tế có 35 câu NFR tổng cộng.
> - Precision = 25/30 = **83%** (trong số dự đoán NFR, 83% đúng)
> - Recall = 25/35 = **71%** (trong số NFR thực, tìm được 71%)
> - F1 = 2 × (0.83 × 0.71) / (0.83 + 0.71) = **76.5%**

#### C. Macro vs. Weighted F1 – Cần Phân Biệt Rõ

| Loại F1 | Cách tính | Khi nào dùng |
|---------|-----------|-------------|
| **Macro F1** | Trung bình đơn giản F1 các class | Khi muốn phản ánh cả class nhỏ (NFR) công bằng |
| **Micro F1** | Gộp TP/FP/FN rồi tính | Bị lệch về class lớn (FR) |
| **Weighted F1** | Trung bình có trọng số theo số mẫu | **Thường báo cáo nhất** |

> ⚠️ **Lưu ý quan trọng**: Với dataset imbalanced, đừng chỉ báo cáo Accuracy! Mô hình đoán toàn bộ là FR vẫn có thể đạt Accuracy 70%+ nhưng Recall NFR = 0%. **Luôn báo cáo Weighted F1 và per-class F1**.

#### D. Cohen's Kappa – Đánh Giá Chất Lượng Annotation

- Đo mức độ đồng thuận giữa các annotators (hoặc model vs. ground truth), loại bỏ yếu tố may mắn
- Kappa > 0.8 = rất tốt; 0.6–0.8 = tốt; < 0.6 = cần xem lại

---

### 2.2 Các Mô Hình AI Phù Hợp Để Thử Nghiệm

#### 🔵 Tier 1 – Classical ML (Baseline – Nhanh, Không Cần GPU)

| Mô hình | Ưu điểm | Nhược điểm | Tool |
|---------|---------|------------|------|
| **Naive Bayes** | Rất nhanh, cần ít data | Giả định độc lập từng từ | scikit-learn |
| **SVM (Linear + TF-IDF)** | Hiệu quả cao với text ngắn | Cần feature engineering | scikit-learn |
| **Logistic Regression** | Interpretable, nhanh | Tuyến tính | scikit-learn |
| **Random Forest** | Robust, ít overfitting | Không hiểu ngữ nghĩa | scikit-learn |

```python
# Ví dụ: SVM Pipeline với TF-IDF
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
    ('svm',   SVC(kernel='linear', class_weight='balanced', probability=True))
])
pipeline.fit(X_train, y_train)
```

#### 🟡 Tier 2 – Transformer-Based Models (State-of-the-art trên dataset nhỏ)

| Mô hình | Params | Đặc điểm | Phù hợp task này |
|---------|--------|----------|-----------------|
| **BERT-base-uncased** | 110M | Bidirectional context | ✅ Tốt |
| **RoBERTa-base** | 125M | BERT cải tiến, train lâu hơn | ✅ Tốt hơn BERT |
| **DistilBERT** | 66M | Nhanh hơn 60%, kém ~3% | ✅ Khi cần tốc độ |
| **DeBERTa-base** | 140M | SOTA cho NLU tasks | ✅ Tốt nhất nhóm này |
| **SciBERT** | 110M | Train trên scientific papers | ✅ Nếu dùng academic reqs |

```python
# Fine-tuning BERT với HuggingFace Transformers
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from datasets import Dataset

model_name = "bert-base-uncased"
tokenizer  = BertTokenizer.from_pretrained(model_name)
model      = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

def tokenize_fn(examples):
    return tokenizer(examples['text'], padding='max_length',
                     truncation=True, max_length=128)

train_ds = Dataset.from_dict({'text': X_train.tolist(), 'label': y_train.tolist()})
train_ds = train_ds.map(tokenize_fn, batched=True)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    evaluation_strategy="epoch",
    weight_decay=0.01,
    load_best_model_at_end=True,
)
trainer = Trainer(model=model, args=training_args, train_dataset=train_ds)
trainer.train()
```

#### 🔴 Tier 3 – Large Language Models (LLMs – Cutting Edge)

| Mô hình | Cách tiếp cận | Chi phí | Khuyến nghị |
|---------|--------------|---------|------------|
| **GPT-4o** | API + Prompt Engineering | ~$0.005/1K tokens | ✅ Zero-shot tốt nhất |
| **Llama 3 (8B/70B)** | Local / HuggingFace Inference | Free (cần GPU) | ✅ Open-source thay thế |
| **Mistral 7B** | Local, rất nhẹ | Free | ✅ Baseline cho open LLM |
| **Claude 3.5 Sonnet** | API | Có phí | ✅ Giải thích tốt |
| **Gemini Pro** | API | Có phí | ✅ Multimodal nếu cần |

**Prompt Engineering cho Classification (ví dụ Zero-shot):**

```python
import openai

SYSTEM_PROMPT = """You are an expert software requirements analyst with 10+ years of experience.

Your task: Classify each software requirement as FR or NFR.
- FR (Functional Requirement): describes WHAT the system does (behavior, features, functions)
- NFR (Non-Functional Requirement): describes HOW WELL the system performs (quality attributes)

Respond ONLY with valid JSON. No extra text.
Format: {"label": "FR" or "NFR", "subcategory": "N/A or Performance/Security/Usability/...", "confidence": 0.0-1.0, "reasoning": "one sentence"}"""

def classify_zero_shot(requirement_text):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Requirement: {requirement_text}"}
        ],
        temperature=0,   # Bắt buộc = 0 cho classification để reproducible
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content
```

**Few-shot Prompting (thêm ví dụ vào prompt):**

```
Classify the following requirement. Here are examples:

EXAMPLE 1:
Input: "The system shall allow administrators to create and manage user accounts."
Output: {"label": "FR"}

EXAMPLE 2:
Input: "The system shall complete all database queries within 500 milliseconds."
Output: {"label": "NFR", "subcategory": "Performance"}

EXAMPLE 3:
Input: "All data transmission shall be encrypted using TLS 1.3."
Output: {"label": "NFR", "subcategory": "Security"}

Now classify:
Input: "{YOUR_REQUIREMENT_HERE}"
```

> 💡 **Mẹo LLM**: Thử **Chain-of-Thought (CoT)** – yêu cầu LLM giải thích lý do TRƯỚC rồi đưa ra nhãn. Thử **Self-consistency** – gọi 5 lần, lấy kết quả đa số. Cả hai thường cải thiện F1 thêm 2–5%.

---

## 🗺️ PHẦN 3: LỘ TRÌNH NGHIÊN CỨU TỪNG BƯỚC

### ⚙️ Bước 0: Chuẩn Bị Nền Tảng (Tuần 1–3)

**Kiến thức cần có trước:**
- [ ] Python cơ bản: list, dict, pandas, numpy
- [ ] ML cơ bản: hiểu train/test split, overfitting, cross-validation
- [ ] NLP cơ bản: tokenization, TF-IDF, word embeddings
- [ ] Đọc ít nhất 5 papers theo hướng dẫn Phần 1

**Cài đặt môi trường:**

```bash
# Tạo môi trường ảo
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Cài packages
pip install pandas numpy scikit-learn matplotlib seaborn
pip install transformers datasets torch torchvision
pip install imbalanced-learn  # xử lý imbalanced data
pip install openai anthropic  # nếu dùng LLM API
pip install nltk spacy
pip install mlflow wandb     # experiment tracking
```

**Platform đề xuất (miễn phí):**
- **Google Colab** – GPU miễn phí (T4) cho BERT fine-tuning
- **Kaggle Notebooks** – GPU miễn phí + datasets tích hợp
- **HuggingFace Spaces** – deploy model miễn phí

---

### 📦 Bước 1: Thu Thập và Chuẩn Bị Dataset

#### Dataset Chính – PROMISE NFR Repository

Đây là dataset **phổ biến nhất** cho bài toán này, được dùng trong hơn 50 papers:

- **Nguồn**: `http://openscience.us/repo/requirements/nfr.html`
- **Kích thước**: 625 câu yêu cầu phần mềm
- **Nhãn**: FR + 11 loại NFR (xem bảng ở Phần 0.2)
- **Định dạng**: CSV, Excel

**Cấu trúc dữ liệu PROMISE (ví dụ):**

```
ProjectID | RequirementText                                           | Class
----------|----------------------------------------------------------|------
1         | The system shall allow users to log in via email          | F
1         | Response time shall not exceed 2 seconds for all queries  | PE
2         | All stored user data shall be encrypted using AES-256     | SE
3         | The interface shall comply with WCAG 2.1 accessibility    | US
4         | The system shall maintain 99.9% uptime per month          | A
```

**Code tải và khám phá dataset:**

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('nfr.csv', encoding='latin-1')
print(df.head())
print(f"\nShape: {df.shape}")

# Phân phối nhãn
print("\nLabel distribution:")
print(df['class'].value_counts())

# Visualize
df['class'].value_counts().plot(kind='bar', figsize=(10,5), color='steelblue')
plt.title('Distribution of Requirement Types (PROMISE Dataset)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('label_distribution.png')
plt.show()

# Binary labels: FR (0) vs NFR (1)
df['binary_label'] = df['class'].apply(lambda x: 0 if x.strip() == 'F' else 1)
print(f"\nFR count: {(df['binary_label']==0).sum()}")
print(f"NFR count: {(df['binary_label']==1).sum()}")
```

#### Các Dataset Bổ Sung

| Dataset | Mô tả | Số lượng | Nguồn |
|---------|-------|---------|-------|
| **PURE** | 79 tài liệu requirements thực tế từ nhiều dự án | ~2,600 reqs | IEEE Dataport |
| **ReqBench** | Benchmark tổng hợp | Varies | Zenodo |
| **GitHub Issues** | Issues từ open-source projects | Tự thu thập | GitHub API |
| **Jira Datasets** | Tự thu thập từ public Jira instances | Tự thu thập | Jira API |

**Cách tự thu thập từ GitHub API:**

```python
import requests

def get_github_issues(owner, repo, token, labels=None):
    headers = {"Authorization": f"token {token}"}
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    params = {"state": "all", "per_page": 100, "labels": labels or ""}
    
    issues = []
    while url:
        response = requests.get(url, headers=headers, params=params)
        issues.extend(response.json())
        url = response.links.get('next', {}).get('url')  # pagination
    
    return [{"title": i["title"], "body": i["body"], "labels": i["labels"]} 
            for i in issues if not i.get("pull_request")]

# Ví dụ: thu thập issues từ một repo lớn
issues = get_github_issues("tensorflow", "tensorflow", token="YOUR_GITHUB_TOKEN")
```

> ⚠️ **Khi tự thu thập dữ liệu**: Cần gán nhãn (annotation) thủ công bằng công cụ như **Label Studio**. Yêu cầu ít nhất 2 annotators. Tính **Inter-Annotator Agreement (IAA)** bằng Cohen's Kappa – kết quả > 0.6 là chấp nhận được để đưa vào nghiên cứu.

---

### 🔧 Bước 2: Tiền Xử Lý Dữ Liệu (Data Preprocessing)

```python
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

nltk.download(['stopwords', 'wordnet', 'punkt'])

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# ─── Giữ lại một số stopwords quan trọng cho requirements ───
# "not", "no", "shall", "must" rất quan trọng – không xóa!
keep_words = {"not", "no", "shall", "must", "should", "will", "cannot"}
stop_words = stop_words - keep_words

def preprocess(text):
    # 1. Lowercase
    text = text.lower()
    # 2. Xóa ký tự đặc biệt (giữ chữ và khoảng trắng)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # 3. Tokenize
    tokens = text.split()
    # 4. Xóa stopwords + Lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return ' '.join(tokens)

df['processed_text'] = df['RequirementText'].apply(preprocess)

# ─── Xử lý Imbalanced Data ───
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer

# Option 1: class_weight='balanced' (đơn giản nhất)
# → truyền vào model: SVC(class_weight='balanced')

# Option 2: Oversampling với SMOTE (trên feature space)
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(df['processed_text'])
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_vec, df['binary_label'])

# ─── Split Dataset ───
X_train, X_test, y_train, y_test = train_test_split(
    df['processed_text'], df['binary_label'],
    test_size=0.2, random_state=42,
    stratify=df['binary_label']    # Giữ tỷ lệ class trong cả train và test
)

print(f"Train: {len(X_train)} samples")
print(f"Test:  {len(X_test)} samples")
print(f"Train label dist: {y_train.value_counts().to_dict()}")
```

**Pipeline NLP Tổng Quát:**

```
Raw Text
  → Lowercase
  → Remove special characters
  → Tokenize
  → Remove stopwords (giữ "not", "shall", "must")
  → Lemmatize (run→run, running→run, requirements→requirement)
  → Feature Extraction:
       ├── TF-IDF Vectors (cho Classical ML)
       ├── Word2Vec / GloVe Embeddings (cho LSTM)
       └── BERT Tokenizer (cho Transformer models)
```

---

### 🤖 Bước 3: Huấn Luyện / Fine-tuning / Prompting Mô Hình

#### Approach A: Classical ML – Xây Dựng Baseline Nhanh

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import cross_val_score

MODELS = {
    "Naive Bayes":         ComplementNB(),
    "SVM (Linear)":        SVC(kernel='linear', class_weight='balanced', probability=True),
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
}

results = {}
for name, clf in MODELS.items():
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
        ('clf',   clf)
    ])
    # 10-fold cross validation
    cv_scores = cross_val_score(pipe, df['processed_text'], df['binary_label'],
                                 cv=10, scoring='f1_weighted')
    results[name] = {"mean": cv_scores.mean(), "std": cv_scores.std()}
    print(f"{name}: F1 = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Train và test
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['FR', 'NFR']))
```

#### Approach B: BERT Fine-tuning – State-of-the-art

```python
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, EarlyStoppingCallback)
from datasets import Dataset
import numpy as np
from sklearn.metrics import f1_score, classification_report

MODEL_NAME = "roberta-base"  # Hoặc "bert-base-uncased", "distilbert-base-uncased"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
model      = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Tokenize
def tokenize_fn(examples):
    return tokenizer(examples['text'], padding='max_length',
                     truncation=True, max_length=128)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"f1_weighted": f1_score(labels, preds, average='weighted'),
            "f1_macro":    f1_score(labels, preds, average='macro')}

# Tạo dataset
train_ds = Dataset.from_dict({'text': X_train.tolist(), 'label': y_train.tolist()})
test_ds  = Dataset.from_dict({'text': X_test.tolist(),  'label': y_test.tolist()})
train_ds = train_ds.map(tokenize_fn, batched=True)
test_ds  = test_ds.map(tokenize_fn,  batched=True)

training_args = TrainingArguments(
    output_dir           = './bert_results',
    num_train_epochs     = 5,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size  = 32,
    learning_rate        = 2e-5,
    warmup_ratio         = 0.1,
    weight_decay         = 0.01,
    evaluation_strategy  = "epoch",
    save_strategy        = "epoch",
    load_best_model_at_end = True,
    metric_for_best_model  = "f1_weighted",
    report_to            = "none",
)

trainer = Trainer(
    model          = model,
    args           = training_args,
    train_dataset  = train_ds,
    eval_dataset   = test_ds,
    compute_metrics= compute_metrics,
    callbacks      = [EarlyStoppingCallback(early_stopping_patience=2)],
)
trainer.train()
```

#### Approach C: LLM Batch Evaluation

```python
import openai, json, time
from tqdm import tqdm

def classify_batch_llm(requirements: list, model="gpt-4o") -> list:
    results = []
    
    for req in tqdm(requirements):
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},  # từ Phần 2
                    {"role": "user",   "content": f"Requirement: {req}"}
                ],
                temperature=0,
                response_format={"type": "json_object"},
                timeout=30
            )
            result = json.loads(response.choices[0].message.content)
            results.append(result.get("label", "UNKNOWN"))
            time.sleep(0.5)  # Rate limit
        except Exception as e:
            print(f"Error: {e}")
            results.append("ERROR")
    
    return results

# Chạy trên test set
llm_predictions = classify_batch_llm(X_test.tolist())
```

---

### 📊 Bước 4: Đánh Giá và Phân Tích Kết Quả

#### 4.1 Evaluation Pipeline Toàn Diện

```python
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, matthews_corrcoef, cohen_kappa_score)
import matplotlib.pyplot as plt
import seaborn as sns

def full_evaluation(y_true, y_pred, y_prob=None, model_name="Model"):
    print(f"\n{'='*55}")
    print(f"  EVALUATION: {model_name}")
    print(f"{'='*55}")
    
    # Per-class report
    print(classification_report(y_true, y_pred, target_names=['FR', 'NFR']))
    
    # Extra metrics
    mcc   = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    print(f"Matthews Corr Coefficient (MCC): {mcc:.4f}")
    print(f"Cohen's Kappa:                   {kappa:.4f}")
    
    if y_prob is not None:
        auc = roc_auc_score(y_true, y_prob)
        print(f"AUC-ROC:                         {auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Pred: FR', 'Pred: NFR'],
                yticklabels=['Actual: FR', 'Actual: NFR'])
    ax.set_title(f'Confusion Matrix – {model_name}', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'cm_{model_name.replace(" ","_")}.png', dpi=150)
    plt.show()
```

#### 4.2 Bảng So Sánh Kết Quả Tổng Hợp (Template Báo Cáo)

| Model | Precision | Recall | F1 (FR) | F1 (NFR) | Weighted F1 | AUC-ROC |
|-------|-----------|--------|---------|---------|-------------|---------|
| Naive Bayes | 0.71 | 0.69 | 0.73 | 0.60 | 0.70 | 0.75 |
| SVM (TF-IDF) | 0.80 | 0.78 | 0.83 | 0.72 | 0.79 | 0.85 |
| Logistic Reg | 0.78 | 0.76 | 0.81 | 0.69 | 0.77 | 0.83 |
| BERT | 0.91 | 0.89 | 0.92 | 0.86 | 0.90 | 0.95 |
| RoBERTa | **0.93** | 0.91 | **0.94** | 0.88 | **0.92** | **0.97** |
| GPT-4o (0-shot) | 0.87 | 0.85 | 0.89 | 0.81 | 0.87 | N/A |
| GPT-4o (few-shot) | 0.91 | **0.92** | 0.92 | **0.90** | 0.91 | N/A |

> *(Các số trên là ví dụ minh họa; kết quả thực tế sẽ khác tùy dataset và cách tinh chỉnh)*

#### 4.3 Error Analysis – Bước Quan Trọng Thường Bị Bỏ Qua!

```python
# Tìm và phân tích lỗi phân loại
import pandas as pd

error_df = pd.DataFrame({
    'requirement': X_test.tolist(),
    'true_label':  ['FR' if l == 0 else 'NFR' for l in y_test],
    'predicted':   ['FR' if p == 0 else 'NFR' for p in y_pred]
})

errors = error_df[error_df['true_label'] != error_df['predicted']]

print("\n=== FALSE POSITIVES: Thực ra là FR nhưng dự đoán là NFR ===")
for _, row in errors[errors['predicted'] == 'NFR'].head(10).iterrows():
    print(f"  [{row['true_label']}→{row['predicted']}] {row['requirement']}")

print("\n=== FALSE NEGATIVES: Thực ra là NFR nhưng dự đoán là FR ===")
for _, row in errors[errors['predicted'] == 'FR'].head(10).iterrows():
    print(f"  [{row['true_label']}→{row['predicted']}] {row['requirement']}")
```

**Câu hỏi cần đặt ra trong Error Analysis:**
- Loại câu nào thường bị nhầm nhất? (câu ngắn, câu dài, câu có từ "shall"?)
- NFR subcategory nào khó nhất? (Security vs. Legal thường bị nhầm nhau)
- Mô hình có bias với class nào? (thường bias về FR – class đa số)

#### 4.4 Statistical Significance Testing – Chứng Minh Mô Hình A Tốt Hơn Mô Hình B

```python
from scipy.stats import wilcoxon, ttest_rel
from sklearn.model_selection import cross_val_score

# Chạy 10-fold CV cho 2 mô hình
scores_svm   = cross_val_score(pipe_svm,  X, y, cv=10, scoring='f1_weighted')
scores_bert  = cross_val_score(pipe_bert, X, y, cv=10, scoring='f1_weighted')

# Wilcoxon signed-rank test (không giả định phân phối chuẩn)
stat, p_value = wilcoxon(scores_svm, scores_bert)
print(f"Wilcoxon test p-value: {p_value:.4f}")
print("→ Khác biệt có ý nghĩa thống kê!" if p_value < 0.05 else "→ Không có ý nghĩa thống kê")
```

---

## 📖 PHẦN 4: TÀI LIỆU VÀ NGUỒN THAM KHẢO

### 4.1 Từ Khóa Tìm Kiếm Papers

**Dùng trên Google Scholar, IEEE Xplore, ACM Digital Library, Scopus:**

| Nhóm chủ đề | Từ khóa tìm kiếm |
|-------------|-----------------|
| Core topic | `"non-functional requirements" classification NLP` |
| Machine Learning | `"requirements classification" "machine learning" accuracy` |
| BERT/Transformers | `"software requirements" BERT transformer classification` |
| LLM approach | `"requirements engineering" GPT LLM classification 2023 2024` |
| Survey papers | `"requirements classification" survey "systematic literature review"` |
| Dataset papers | `"NFR dataset" "requirements engineering" benchmark` |
| Evaluation | `"requirements classification" "F1-score" precision recall` |
| LLM + prompting | `"prompt engineering" "software requirements" classification` |

**Filter thêm**: Year ≥ 2021 để lấy nghiên cứu cập nhật nhất với deep learning và LLMs.

**Venue tốt để tìm papers trong lĩnh vực này:**
- **IEEE**: IEEE Transactions on Software Engineering (TSE), ICSE, ICSME
- **ACM**: ESEC/FSE, MSR, RE (Requirements Engineering Conference)
- **Journals**: Information and Software Technology (IST), Empirical Software Engineering (EMSE)
- **arXiv**: cs.SE, cs.CL (tìm preprints mới nhất)

---

### 4.2 Seminal Papers Nên Đọc (Theo Thứ Tự)

**Vòng 1 – Nền tảng (đọc đầu tiên):**

1. Cleland-Huang, J., Settimi, R., et al. (2006). "Automated classification of non-functional requirements." – *Requirements Engineering journal*. Đây là paper gốc định nghĩa bài toán.

2. Slankas, J., & Williams, L. (2013). "Automated extraction and classification of non-functional requirements from plain text." – *RE Conference*. Dùng SVM + NLP pipeline cổ điển.

**Vòng 2 – Thời kỳ Deep Learning (2018–2021):**

3. Tìm papers với keyword: `BERT "software requirements" classification 2019 2020 2021`

4. Tìm papers với keyword: `"transfer learning" "non-functional requirements" classification`

**Vòng 3 – Thời kỳ LLM (2022–2025):**

5. Tìm papers: `GPT-4 "requirements engineering" classification accuracy 2023`

6. Tìm papers: `"large language models" "software requirements" classification evaluation`

---

### 4.3 Public Datasets – Bắt Tay Thực Hành Ngay

| Dataset | Mô tả chi tiết | Link tìm | Format |
|---------|---------------|----------|--------|
| **PROMISE NFR** | 625 requirements, 12 classes (FR + 11 NFR types) – **dataset chuẩn nhất** | `openscience.us/repo/requirements/nfr.html` | CSV |
| **PURE** | 79 real-world requirement documents từ nhiều domain | IEEE Dataport → search "PURE requirements" | JSON + PDF |
| **ReqView Samples** | Tài liệu requirements mẫu nhỏ | `reqview.com/examples` | HTML |
| **Zenodo RE Datasets** | Nhiều datasets đa dạng do researchers publish | `zenodo.org` → search "software requirements classification" | Varies |

**Cách tìm datasets trên Papers With Code:**
- Vào `paperswithcode.com`
- Search: "requirements classification"
- Tab "Datasets" → xem benchmark datasets được dùng nhiều nhất

**Cách tìm datasets trên Kaggle:**
```
Kaggle search: "functional requirements classification"
Kaggle search: "software requirements NLP dataset"
Kaggle search: "NFR requirements engineering"
```

**Cách tìm datasets trên Hugging Face Datasets Hub:**
```python
from datasets import load_dataset, list_datasets

# Tìm kiếm
matching = [d for d in list_datasets() if 'requirement' in d.lower()]
print(matching)
```

---

### 4.4 Tools & Platforms Tổng Hợp

| Mục đích | Tool | Link | Miễn phí? |
|---------|------|------|-----------|
| Viết code ML (có GPU) | **Google Colab** | colab.research.google.com | ✅ T4 GPU free |
| Viết code ML (có GPU) | **Kaggle Notebooks** | kaggle.com | ✅ P100 GPU free |
| Quản lý papers | **Zotero** | zotero.org | ✅ |
| Ghi chú nghiên cứu | **Obsidian / Notion** | obsidian.md | ✅ |
| Pretrained models | **HuggingFace Hub** | huggingface.co | ✅ |
| Annotation tool | **Label Studio** | labelstud.io | ✅ Open-source |
| Experiment tracking | **Weights & Biases** | wandb.ai | ✅ Free tier |
| Experiment tracking | **MLflow** | mlflow.org | ✅ |
| Đọc arxiv papers | **Semantic Scholar** | semanticscholar.org | ✅ |
| Tìm papers related | **Connected Papers** | connectedpapers.com | ✅ |

---

## 🗓️ PHẦN 5: TIMELINE NGHIÊN CỨU TỔNG QUAN

| Tuần | Mục tiêu chính | Output cụ thể |
|------|---------------|---------------|
| **1–2** | Đọc 5–7 survey papers, nắm vững FR/NFR | Bảng so sánh papers, ghi chú |
| **3–4** | Đọc thêm 8–10 specific papers, xác định gap | Research gap list + câu hỏi nghiên cứu |
| **5** | Download PROMISE, EDA, thống kê | Jupyter notebook với biểu đồ phân phối |
| **6** | Xây Baseline: Naive Bayes + SVM | Baseline results table |
| **7–8** | Fine-tune BERT hoặc RoBERTa | BERT results + confusion matrix |
| **9–10** | LLM experiments: GPT-4o / Llama 3 (zero/few-shot) | LLM results comparison |
| **11** | Error analysis sâu, phân tích lỗi từng model | Error analysis report |
| **12** | Tổng hợp, viết báo cáo, kết luận | Final comparison table + conclusion |

---

## ⚠️ 10 Lưu Ý Quan Trọng Cho Người Mới

1. **Đừng bỏ qua Baseline**: Luôn bắt đầu với SVM/Naive Bayes trước BERT. Nếu BERT chỉ tốt hơn 1% → nghiên cứu sâu thêm về tại sao
2. **Dataset nhỏ = Overfitting nguy hiểm**: PROMISE chỉ có 625 rows! Luôn dùng 10-fold cross-validation
3. **F1 quan trọng hơn Accuracy**: Với imbalanced data, Accuracy 80% có thể che giấu việc mô hình bỏ sót toàn bộ NFR
4. **Reproducibility là bắt buộc**: Set `random_state=42` ở mọi nơi, log parameters, lưu model weights
5. **Error Analysis là cốt lõi**: Đọc kỹ các câu bị dự đoán sai – đây là chìa khóa để cải thiện
6. **Bắt đầu từ dataset có sẵn**: Đừng lãng phí tuần 1 vào tự thu thập dữ liệu; dùng PROMISE trước
7. **Prompt LLM phải nghiêm túc**: temperature=0 cho mọi classification task; thử nhiều prompt templates
8. **Đọc kỹ Limitations trong papers**: Đây là nguồn research gap phong phú nhất, đừng bỏ qua
9. **NFR khó hơn FR nhiều**: F1 cho class NFR thường thấp hơn FR 10–15%. Đây là điểm cần phân tích
10. **Communicate kết quả rõ ràng**: Luôn kèm confidence intervals, số lượng mẫu, và môi trường thực nghiệm

---

*Chúc bạn nghiên cứu thành công! 🚀 Remember: "A good researcher is not one who knows all the answers, but one who asks the right questions."*
