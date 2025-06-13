# SVM Text Classification untuk Deteksi URL Berbahaya
# Python 3.9 Compatible

# Import library yang diperlukan
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')

print("Library berhasil diimport!")

# Baca data dari file Excel
print("Membaca data dari datatrain.xlsx...")
try:
    df = pd.read_excel('datatrain.xlsx')
    print(f"Data berhasil dibaca! Shape: {df.shape}")
    print("\nInfo dataset:")
    print(df.info())
    print("\nSample data:")
    print(df.head())
except FileNotFoundError:
    print("File datatrain.xlsx tidak ditemukan. Pastikan file berada di direktori yang sama.")
except Exception as e:
    print(f"Error membaca file: {e}")

# Eksplorasi data
print("\n=== EKSPLORASI DATA ===")
print(f"Jumlah data: {len(df)}")
print(f"Kolom yang tersedia: {df.columns.tolist()}")
print("\nDistribusi label:")
print(df['label'].value_counts())
print("\nPersentase label:")
print(df['label'].value_counts(normalize=True) * 100)

# Cek missing values
print("\nMissing values:")
print(df.isnull().sum())

# Visualisasi distribusi label
plt.figure(figsize=(8, 6))
df['label'].value_counts().plot(kind='bar', color=['lightcoral', 'lightblue'])
plt.title('Distribusi Label Dataset')
plt.xlabel('Label')
plt.ylabel('Jumlah')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Preprocessing text
def preprocess_text(text):
    """
    Fungsi untuk preprocessing teks
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters and numbers (opsional)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespaces
    text = ' '.join(text.split())
    
    return text

print("\n=== PREPROCESSING DATA ===")
# Cek dan hapus missing values terlebih dahulu
print(f"Missing values sebelum preprocessing:")
print(f"text: {df['text'].isnull().sum()}")
print(f"label: {df['label'].isnull().sum()}")

# Hapus rows dengan missing values di kolom text atau label
df = df.dropna(subset=['text', 'label'])
print(f"Jumlah data setelah menghapus missing values: {len(df)}")

# Apply preprocessing
df['text_processed'] = df['text'].apply(preprocess_text)

# Remove rows with empty text setelah preprocessing
df = df[df['text_processed'].str.len() > 0]
print(f"Jumlah data setelah preprocessing: {len(df)}")

# Reset index setelah filtering
df = df.reset_index(drop=True)

# Prepare features and target
X = df['text_processed']
y = df['label']

# Pastikan tidak ada NaN di X dan y
print(f"Missing values setelah preprocessing:")
print(f"X (text): {X.isnull().sum()}")
print(f"y (label): {y.isnull().sum()}")

print(f"Jumlah sampel: {len(X)}")
print(f"Distribusi label setelah preprocessing:")
print(y.value_counts())

# Cek jika masih ada nilai kosong
if X.isnull().any() or y.isnull().any():
    print("PERINGATAN: Masih ada nilai NaN, menghapus rows dengan NaN...")
    valid_indices = ~(X.isnull() | y.isnull())
    X = X[valid_indices]
    y = y[valid_indices]
    print(f"Jumlah data final: {len(X)}")

# Pastikan ada data yang cukup untuk training
if len(X) < 10:
    print("ERROR: Data terlalu sedikit untuk training!")
else:
    print("Data siap untuk training!")

# Split data train-test
print("\n=== SPLIT DATA ===")

# Validasi final sebelum split
print("Validasi data sebelum split:")
print(f"Tipe data X: {type(X)}")
print(f"Tipe data y: {type(y)}")
print(f"Unique labels: {y.unique()}")
print(f"Label counts: {y.value_counts().to_dict()}")

# Pastikan ada minimal 2 class untuk stratify
if len(y.unique()) < 2:
    print("ERROR: Hanya ada 1 class, tidak bisa melakukan stratified split!")
    print("Menggunakan split biasa tanpa stratify...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42
    )
else:
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            stratify=y, 
            random_state=42
        )
    except Exception as e:
        print(f"Stratified split gagal: {e}")
        print("Menggunakan split biasa...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42
        )

print(f"Data training: {len(X_train)} sampel")
print(f"Data testing: {len(X_test)} sampel")
print(f"Distribusi label training: {y_train.value_counts().to_dict()}")
print(f"Distribusi label testing: {y_test.value_counts().to_dict()}")

# Feature extraction menggunakan TF-IDF
print("\n=== FEATURE EXTRACTION ===")
# Inisialisasi TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    max_features=5000,  # Maksimal 5000 fitur
    min_df=2,          # Minimal muncul di 2 dokumen
    max_df=0.8,        # Maksimal muncul di 80% dokumen
    stop_words='english',  # Hapus stop words bahasa Inggris
    ngram_range=(1, 2)     # Unigram dan bigram
)

# Fit dan transform data training
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"Shape X_train_tfidf: {X_train_tfidf.shape}")
print(f"Shape X_test_tfidf: {X_test_tfidf.shape}")
print(f"Jumlah fitur TF-IDF: {len(tfidf.get_feature_names_out())}")

# Training SVM Model
print("\n=== TRAINING SVM MODEL ===")
# Inisialisasi SVM classifier
svm_model = SVC(
    kernel='linear',    # Kernel linear untuk text classification
    C=1.0,             # Regularization parameter
    random_state=42,
    probability=True   # Untuk mendapatkan probabilitas prediksi
)

# Training model
print("Memulai training...")
svm_model.fit(X_train_tfidf, y_train)
print("Training selesai!")

# Prediksi pada data test
print("\n=== EVALUASI MODEL ===")
y_pred = svm_model.predict(X_test_tfidf)
y_pred_proba = svm_model.predict_proba(X_test_tfidf)

# Evaluasi performa
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualisasi Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=svm_model.classes_, 
            yticklabels=svm_model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

import joblib
from datetime import datetime

# Simpan model dan vectorizer ke dalam file .pkl
print("\n=== MENYIMPAN MODEL ===")

# Membuat nama file dengan timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f'svm_model_{timestamp}.pkl'
vectorizer_filename = f'tfidf_vectorizer_{timestamp}.pkl'

try:
    # Simpan model SVM
    joblib.dump(svm_model, model_filename)
    print(f"Model SVM berhasil disimpan sebagai {model_filename}")
    
    # Simpan TF-IDF Vectorizer
    joblib.dump(tfidf, vectorizer_filename)
    print(f"Vectorizer berhasil disimpan sebagai {vectorizer_filename}")
    
    # Simpan juga dalam file dengan nama tetap (untuk kemudahan loading)
    joblib.dump(svm_model, 'svm_model.pkl')
    joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
    print("Model juga disimpan dengan nama tetap 'svm_model.pkl' dan 'tfidf_vectorizer.pkl'")
    
except Exception as e:
    print(f"Gagal menyimpan model: {e}")