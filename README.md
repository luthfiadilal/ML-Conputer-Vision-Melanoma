# 🩺 Klasifikasi Kanker Kulit Melanoma Menggunakan MobileNetV2

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![Architecture](https://img.shields.io/badge/Model-MobileNetV2-blue)](#-arsitektur-model)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-85.4%25-success)](#-hasil-evaluasi)

Repository ini berisi proyek *Computer Vision* dan *Deep Learning* untuk mendeteksi dan mengklasifikasikan gambar lesi kulit guna membantu deteksi dini **Melanoma** menggunakan arsitektur **MobileNetV2** dengan pendekatan *Transfer Learning*.

---

## 📌 Daftar Isi
- [Latar Belakang & Deskripsi](#-latar-belakang--deskripsi)
- [Kelas Target](#-kelas-target)
- [Dataset & Pra-pemrosesan](#-dataset--pra-pemrosesan)
- [Arsitektur Model](#-arsitektur-model)
- [Hasil Evaluasi & Kinerja](#-hasil-evaluasi--kinerja)
- [Struktur Direktori](#-struktur-direktori)
- [Panduan Instalasi & Penggunaan](#-panduan-instalasi--penggunaan)
- [Contoh Inferensi (Pengujian Gambar Baru)](#-contoh-inferensi-pengujian-gambar-baru)
- [Rencana Pengembangan](#-rencana-pengembangan-roadmap)
- [Kontributor](#-kontributor)

---

## 📖 Latar Belakang & Deskripsi

Melanoma adalah jenis kanker kulit yang paling berbahaya dan berpotensi mematikan jika tidak terdeteksi pada stadium awal. Pendekatan berbasis *Deep Learning* dan *Computer Vision* dapat memberikan dukungan *computer-aided diagnosis* (CAD) bagi tenaga medis maupun masyarakat umum untuk penyaringan awal lesi kulit.

Proyek ini melatih model klasifikasi gambar yang efisien dan ringan menggunakan **MobileNetV2**. Selain membedakan lesi melanoma dan non-melanoma, model ini juga dilengkapi dengan kelas **NonSkin** agar mampu mengidentifikasi dan mengabaikan input gambar yang bukan merupakan permukaan kulit manusia.

---

## 🏷️ Kelas Target

Model mengklasifikasikan citra ke dalam **3 kelas**:

| Label Index | Nama Kelas | Deskripsi |
|:---:|:---|:---|
| `0` | **Melanoma** | Citra lesi kulit terindikasi kanker melanoma ganas. |
| `1` | **NonSkin** | Citra bukan kulit (objek lain, pakaian, background) untuk mencegah kesalahan prediksi jika pengguna mengunggah gambar selain kulit. |
| `2` | **NotMelanoma** | Citra kulit normal atau lesi jinak (*benign* / tahi lalat biasa / non-melanoma). |

---

## 📊 Dataset & Pra-pemrosesan

### Pembagian Data (Data Splitting)
Total dataset terdiri dari **3.339 gambar**, dibagi dengan rasio:
- **Training Set (70%)**: 2.336 gambar
- **Validation Set (20%)**: 666 gambar
- **Test Set (10%)**: 336 gambar (seimbang: 112 gambar per kelas)

### Augmentasi Data
Untuk meningkatkan generalisasi model dan mengatasi *overfitting*, data latih diberi augmentasi secara *real-time*:
- **Target Size**: 224 x 224 piksel
- **Batch Size**: 32
- **Rotation Range**: 15 derajat
- **Width Shift Range**: 0.1
- **Height Shift Range**: 0.1
- **Zoom Range**: 0.1
- **Horizontal Flip**: Aktif (`True`)
- **Fill Mode**: `nearest`
- **Preprocessing**: Fungsi pra-pemrosesan bawaan MobileNetV2 (skala nilai piksel ke rentang `[-1, 1]`)

---

## 🧠 Arsitektur Model

Proyek ini memanfaatkan teknik **Transfer Learning**:

```
Input Image (224 x 224 x 3)
         │
         ▼
MobileNetV2 Backbone (Pre-trained ImageNet, Frozen)
         │
         ▼
GlobalAveragePooling2D
         │
         ▼
Dropout (0.3)
         │
         ▼
Dense Layer (3 unit, Softmax)
         │
         ▼
Output Prediksi: [Melanoma, NonSkin, NotMelanoma]
```

### Konfigurasi & Hyperparameter
- **Base Model**: MobileNetV2 (`include_top=False`, bobot `imagenet`)
- **Backbone Status**: Dibekukan (*Frozen* / `trainable=False`)
- **Dropout Rate**: 0.3
- **Optimizer**: Adam
- **Loss Function**: `categorical_crossentropy`
- **Metrik Utama**: `accuracy`
- **Total Parameter**: 2.261.827 parameter
  - *Trainable params*: 3.843 parameter (lapisan Dense akhir)
  - *Non-trainable params*: 2.257.984 parameter (MobileNetV2)
- **Early Stopping**: Monitor `val_loss`, `patience=3`, dengan `restore_best_weights=True`

---

## 📈 Hasil Evaluasi & Kinerja

Model diuji pada data uji independen (*test set*) sebanyak **336 gambar**.

### Metrik Pengujian
- **Test Loss**: ~0.33
- **Test Accuracy**: **85.4% – 87.0%**

### Classification Report

```text
              precision    recall  f1-score   support

    Melanoma       0.81      0.78      0.79       112
     NonSkin       1.00      1.00      1.00       112
 NotMelanoma       0.79      0.82      0.80       112

    accuracy                           0.87       336
   macro avg       0.87      0.87      0.87       336
weighted avg       0.87      0.87      0.87       336
```

> [!NOTE]
> - Model memiliki akurasi dan F1-Score sempurna (**100%**) dalam mengenali data **NonSkin**, memastikan sistem tidak salah memprediksi objek asing sebagai lesi kulit.
> - Presisi untuk **Melanoma** mencapai **81%** dengan recall **78%**.

---

## 📁 Struktur Direktori

```text
Melanoma/
├── FIX_Model_MobileNetV2_Melanoma.ipynb   # Notebook utama pelatihan & evaluasi model
├── README.md                             # Dokumentasi proyek
└── (model_melanoma_baru.h5)               # File model tersimpan hasil training
```

---

## 🚀 Panduan Instalasi & Penggunaan

### 1. Prasyarat Sistem
- Python 3.9 atau lebih baru
- Virtual environment (opsional namun disarankan)

### 2. Kloning Repository
```bash
git clone https://github.com/luthfiadilal/ML-Conputer-Vision-Melanoma.git
cd ML-Conputer-Vision-Melanoma
```

### 3. Instalasi Dependensi
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn pillow
```

### 4. Menjalankan Notebook
Anda dapat membuka dan menjalankan file [FIX_Model_MobileNetV2_Melanoma.ipynb](file:///C:/Users/Luthfi/Downloads/Melanoma/FIX_Model_MobileNetV2_Melanoma.ipynb) melalui **Jupyter Lab / Notebook** atau mengunggahnya ke **Google Colab**:

```bash
jupyter notebook FIX_Model_MobileNetV2_Melanoma.ipynb
```

---

## 💡 Contoh Inferensi (Pengujian Gambar Baru)

Berikut contoh skrip Python untuk melakukan prediksi pada sebuah citra menggunakan model yang telah disimpan:

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# 1. Load model yang sudah dilatih
model = tf.keras.models.load_model("model_melanoma_baru.h5")

# 2. Definisikan label kelas sesuai training
class_names = ["Melanoma", "NonSkin", "NotMelanoma"]

# 3. Fungsi prediksi citra
def predict_skin_lesion(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions)) * 100

    return predicted_class, confidence

# Contoh pemanggilan:
# hasil, conf = predict_skin_lesion("contoh_gambar.jpg")
# print(f"Hasil Prediksi : {hasil} ({conf:.2f}%)")
```

---

## 🔮 Rencana Pengembangan (Roadmap)

- [ ] **Fine-Tuning Lanjutan**: Membuka blok *unfreeze* pada *layer-layer* akhir MobileNetV2 untuk meningkatkan metrik *Recall* pada kelas Melanoma.
- [ ] **Konversi Model**: Mengekspor model ke format TensorFlow Lite (`.tflite`) untuk implementasi pada perangkat *mobile* (Android/iOS).
- [ ] **Aplikasi Web / API**: Membangun antarmuka pengguna berbasis Streamlit atau FastAPI untuk pengujian interaktif secara visual.
- [ ] **Explainable AI (XAI)**: Menambahkan visualisasi Grad-CAM untuk melihat area spesifik lesi yang memicu keputusan model.

---

## 👥 Kontributor

Dikembangkan oleh **[Luthfi Adilal Mahbub](https://github.com/luthfiadilal)**.  
Kritik, saran, serta kontribusi pull request sangat terbuka dan dipersilakan!
