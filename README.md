# Credit Card Fraud Detection
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/zehraoztturk/credit-card-fraud-detection)
## English

### Overview

This project compares a simple rule-based baseline system against a Random Forest machine learning model for credit card fraud detection, using the publicly available [Kaggle ULB Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset. The goal is to quantify the measurable advantages of ML-based anomaly detection over static threshold rules in terms of Precision, Recall, F1-Score, and ROC-AUC.



### Dataset

- **Source:** Kaggle — ULB Machine Learning Group (Credit Card Fraud Detection)
- **Size:** 284,807 transactions; 492 fraudulent cases (~0.17%)
- **Features:** V1–V28 (PCA-transformed for confidentiality), Time, Amount, Class (0 = legitimate, 1 = fraud)
- **Missing values:** None

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/z-ozturk/Credit-Card-Fraud-Detection.git
   cd Credit-Card-Fraud-Detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Provide the dataset** — choose one option:
   - **Option A (local):** Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root.
   - **Option B (auto-download):** Configure your [Kaggle API credentials](https://www.kaggle.com/docs/api) (`kaggle.json`). The notebook will detect that the file is missing and download it automatically.

4. **Run the notebook**
   ```bash
   jupyter notebook
   ```
   Open `Credit_Card_Fraud_Detection_Rule_Based_System_vs_Random_Forest.ipynb` and run all cells top-to-bottom.

### Project Structure

| File | Description |
|------|-------------|
| `Credit_Card_Fraud_Detection_Rule_Based_System_vs_Random_Forest.ipynb` | Main Jupyter notebook — data loading, EDA, rule-based system, Random Forest, evaluation |
| `credit_card_fraud_detection_rule_based_system_vs_random_forest.py` | Equivalent Python script (same logic, runnable outside Jupyter) |
| `Proje_Raporu.pdf` | Original project report in Turkish |
| `Project_Report_EN.md` | Full English translation of the project report |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Excludes the large dataset file and generated outputs |

### Results

| Model | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| Rule-Based System (75th-percentile threshold) | 0.0024 | 0.3537 | 0.0049 | 0.5519 |
| Random Forest (`n_estimators=100`, `class_weight='balanced'`) | **0.9605** | **0.7449** | **0.8391** | **0.9529** |

Key takeaways:
- The rule-based system generated **71,028 false positives** vs. only **3** for Random Forest.
- F1-Score improved **~171×** and Precision improved **~400×**.
- Feature importance analysis shows that `Amount` — the sole criterion for the rule-based system — is among the *least* important features for the ML model; the PCA components V14, V10, V12, and V4 dominate.

### Tools Used

Python · pandas · NumPy · scikit-learn · Matplotlib · seaborn · Jupyter · Kaggle API

---

## Türkçe

### Genel Bakış

Bu proje, kamuya açık [Kaggle ULB Kredi Kartı Sahtekarlığı Tespiti](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) veri seti üzerinde basit bir kural tabanlı referans sistemi ile Random Forest makine öğrenmesi modelini karşılaştırmaktadır. Amaç, Precision, Recall, F1-Score ve ROC-AUC metrikleri açısından makine öğrenmesi tabanlı anomali tespitinin statik eşik kurallarına göre sunduğu ölçülebilir avantajları ortaya koymaktır.

### Veri Seti

- **Kaynak:** Kaggle — ULB Makine Öğrenmesi Grubu (Kredi Kartı Sahtekarlığı Tespiti)
- **Boyut:** 284.807 işlem; 492 sahte işlem (~%0,17)
- **Özellikler:** V1–V28 (gizlilik için PCA dönüşümlü), Time, Amount, Class (0 = gerçek, 1 = sahte)
- **Eksik değer:** Yok

### Kurulum

1. **Repoyu klonlayın**
   ```bash
   git clone https://github.com/z-ozturk/Credit-Card-Fraud-Detection.git
   cd Credit-Card-Fraud-Detection
   ```

2. **Bağımlılıkları yükleyin**
   ```bash
   pip install -r requirements.txt
   ```

3. **Veri setini sağlayın** — iki seçenekten birini kullanın:
   - **Seçenek A (yerel):** `creditcard.csv` dosyasını [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)'dan indirin ve proje klasörüne koyun.
   - **Seçenek B (otomatik indirme):** [Kaggle API kimlik bilgilerini](https://www.kaggle.com/docs/api) (`kaggle.json`) yapılandırın. Notebook, dosyanın eksik olduğunu tespit ederek otomatik olarak indirecektir.

4. **Notebook'u çalıştırın**
   ```bash
   jupyter notebook
   ```
   `Credit_Card_Fraud_Detection_Rule_Based_System_vs_Random_Forest.ipynb` dosyasını açın ve tüm hücreleri yukarıdan aşağıya çalıştırın.

### Proje Yapısı

| Dosya | Açıklama |
|-------|----------|
| `Credit_Card_Fraud_Detection_Rule_Based_System_vs_Random_Forest.ipynb` | Ana Jupyter notebook — veri yükleme, EDA, kural tabanlı sistem, Random Forest, değerlendirme |
| `credit_card_fraud_detection_rule_based_system_vs_random_forest.py` | Eşdeğer Python betiği (Jupyter dışında çalıştırılabilir) |
| `Proje_Raporu.pdf` | Türkçe orijinal proje raporu |
| `Project_Report_EN.md` | Proje raporunun tam İngilizce çevirisi |
| `requirements.txt` | Python bağımlılıkları |
| `.gitignore` | Büyük veri seti dosyasını ve üretilen çıktıları dışarıda bırakır |

### Sonuçlar

| Model | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| Kural Tabanlı Sistem (75. yüzdelik eşik) | 0,0024 | 0,3537 | 0,0049 | 0,5519 |
| Random Forest (`n_estimators=100`, `class_weight='balanced'`) | **0,9605** | **0,7449** | **0,8391** | **0,9529** |

Öne çıkan bulgular:
- Kural tabanlı sistem **71.028 yanlış pozitif** üretirken Random Forest yalnızca **3** üretmiştir.
- F1-Score yaklaşık **171 kat**, Precision yaklaşık **400 kat** iyileşmiştir.
- Özellik önemi analizi, kural tabanlı sistemin tek kriteri olan `Amount` değişkeninin makine öğrenmesi modelinde en *az* önemli özellikler arasında yer aldığını göstermektedir; V14, V10, V12 ve V4 PCA bileşenleri baskın çıkmaktadır.

### Kullanılan Araçlar

Python · pandas · NumPy · scikit-learn · Matplotlib · seaborn · Jupyter · Kaggle API
