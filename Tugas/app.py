# app.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

st.title("📊 Decision Tree Classifier - Diabetes Dataset")

# 1. Load dataset
df = pd.read_csv("diabetes.csv")  # kamu bisa simpan dataset ini sebagai diabetes.csv
st.write("### Data Awal", df.head())

# 2. Pisahkan fitur & label
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# 5. Evaluasi
y_pred = clf.predict(X_test)
st.write("### Akurasi:", round(accuracy_score(y_test, y_pred)*100, 2), "%")
st.text("Classification Report:\n" + classification_report(y_test, y_pred))

# 6. Visualisasi
st.write("### Visualisasi Pohon Keputusan")
fig, ax = plt.subplots(figsize=(12,8))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=["Tidak Diabetes","Diabetes"], fontsize=8)
st.pyplot(fig)

# 7. Input manual prediksi
st.write("### Prediksi Input Manual")
inputs = []
for col in X.columns:
    val = st.number_input(f"{col}", value=0.0)
    inputs.append(val)

if st.button("Prediksi"):
    pred = clf.predict([inputs])
    hasil = "Diabetes" if pred[0] == 1 else "Tidak Diabetes"
    st.success(f"Hasil Prediksi: {hasil}")
