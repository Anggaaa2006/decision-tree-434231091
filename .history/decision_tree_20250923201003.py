# ============================================
# 1. Import Library
# ============================================
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import ConfusionMatrixDisplay

# ============================================
# 2. Menampilkan semua data
# ============================================
dataframe = pd.read_excel("BlaBla.xlsx")

# Misal dataset punya 14 kolom (A–N), sesuaikan dengan data aslinya
data = dataframe[['A','B','C','D','E','F','G','H','I','J','K','L','M','N']]

print("data awal".center(75,"="))
print(data)
print("="*75)

# ============================================
# 3. Grouping
# ============================================
print("GROUPING VARIABEL".center(75,"="))
X = data.iloc[:, 0:13].values   # 13 kolom pertama → variabel
y = data.iloc[:, 13].values     # kolom ke-14 → target
print("data variabel".center(75,"="))
print(X)
print("data kelas".center(75,"="))
print(y)
print("="*75)

# ============================================
# 4. Training dan Testing
# ============================================
print("SPLITTING DATA 20-80".center(75,"="))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)
print("instance variabel data training".center(75,"="))
print(X_train)
print("instance kelas data training".center(75,"="))
print(y_train)
print("instance variabel data testing".center(75,"="))
print(X_test)
print("instance kelas data testing".center(75,"="))
print(y_test)
print("="*75)

# ============================================
# 5. Decision Tree
# ============================================
decision_tree = DecisionTreeClassifier(random_state=0)
decision_tree.fit(X_train, y_train)

print("instance prediksi decision tree: ")
Y_pred = decision_tree.predict(X_test)
print(Y_pred)
print("="*75)

# ============================================
# 6. Prediksi Akurasi
# ============================================
accuracy = round(accuracy_score(y_test, Y_pred) * 100, 2)
print("Akurasi: ", accuracy, "%")
print("="*75)

# ============================================
# 7. Classification Report
# ============================================
print("CLASSIFICATION REPORT DECISION TREE".center(75,"="))
print(classification_report(y_test, Y_pred))

cm = confusion_matrix(y_test, Y_pred)
print("Confusion Matrix:")
print(cm)
print("="*75)

# (Opsional: tampilkan confusion matrix grafik)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# ============================================
# 8. Visualisasi Decision Tree
# ============================================
plt.figure(figsize=(18,10))
plot_tree(
    decision_tree,
    filled=True,
    feature_names=['A','B','C','D','E','F','G','H','I','J','K','L','M'],
    class_names=True
)
plt.show()

# ============================================
# 9. Klasifikasi / Prediksi Input
# ============================================
print("CONTOH INPUT".center(75,"="))
A = int(input("Umur Pasien = "))

print("Isi Jenis Kelamin dengan 0 jika Perempuan dan 1 jika Laki-Laki")
B = int(input("Jenis Kelamin Pasien = "))

C = int(input("Apakah pasien mengalami C? (1=Ya,0=Tidak): "))
D = int(input("Apakah pasien mengalami D? (1=Ya,0=Tidak): "))
E = int(input("Apakah pasien mengalami E? (1=Ya,0=Tidak): "))
F = int(input("Apakah pasien mengalami F? (1=Ya,0=Tidak): "))
G = int(input("Apakah pasien mengalami G? (1=Ya,0=Tidak): "))
H = int(input("Apakah pasien mengalami H? (1=Ya,0=Tidak): "))

J = input("Apakah pasien mengalami J? (Y/T): ")
if J == "Y":
    J = 1
else:
    J = 0

K = input("Apakah pasien mengalami K? (Y/T): ")
if K == "Y":
    K = 1
else:
    K = 0

L = input("Apakah pasien mengalami L? (Y/T): ")
if L == "Y":
    L = 1
else:
    L = 0

M = input("Apakah pasien mengalami M? (Y/T): ")
if M == "Y":
    M = 1
else:
    M = 0

# Susun array input sesuai urutan dataset
Train = [A, B, C, D, E, F, G, H, J, K, L, M]
print("Data Input:", Train)

# Convert ke DataFrame
test = pd.DataFrame([Train])
predtest = decision_tree.predict(test)

if predtest == 1:
    print("Pasien Positive")
else:
    print("Pasien Negative")