import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Memuat dataset
df = pd.read_csv("data/Combined Data.csv")

# Menampilkan beberapa baris pertama untuk melihat struktur data
print(df[:1])

# Menghapus baris dengan status 'suicidal' berdasarkan kondisi
df = df.drop(df[df['status'] == 'Suicidal'].index)

# Cek apakah status 'suicidal' sudah dihapus
print(df['status'].value_counts())
df.isnull().sum()
df = df.dropna()
df.isna().sum()

# Inisialisasi lemmatizer dan stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Ubah ke lowercase
    text = text.lower()

    # Tokenisasi
    tokens = word_tokenize(text)

    # Menghapus tanda baca dan stopwords
    tokens = [t for t in tokens if t not in stop_words and t not in string.punctuation]

    # Lemmatization
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)

# Terapkan preprocessing ke kolom 'statement'
df['statement'] = df['statement'].fillna('')
df['processed_statement'] = df['statement'].apply(preprocess_text)

# # Lihat beberapa hasil setelah preprocessing
# print(df[['statement', 'processed_statement']].head())
# # Memisahkan fitur dan label
# X = df['processed_statement']
# y = df['status']
# # Encode label
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)
# joblib.dump(label_encoder, 'label_encoder.pkl')
# # Pisahkan data
# X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
# # Inisialisasi tokenizer
# tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")  # Batasi hingga 5000 kata
# tokenizer.fit_on_texts(X_train)

# # Transformasi teks menjadi sequences
# X_train_seq = tokenizer.texts_to_sequences(X_train)
# X_test_seq = tokenizer.texts_to_sequences(X_test)

# # Padding sequences
# X_train_padded = pad_sequences(X_train_seq, maxlen=100, padding='post')
# X_test_padded = pad_sequences(X_test_seq, maxlen=100, padding='post')

# # Simpan tokenizer
# joblib.dump(tokenizer, 'tokenizer.pkl')
# # Bangun model
# model = Sequential([
#     tf.keras.layers.Embedding(input_dim=5000, output_dim=64, input_length=100),
#     tf.keras.layers.LSTM(64, return_sequences=True),
#     tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.LSTM(32),
#     tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')  # Output layer
# ])
# # Kompilasi model
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(),
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#     metrics=[tf.keras.metrics.Accuracy()]
# )

# model.summary()
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# history = model.fit(
#     X_train_padded,
#     y_train,
#     validation_data=(X_test_padded, y_test),
#     epochs=20,
#     batch_size=32,
#     callbacks=[early_stopping]
# )

# # Evaluasi model
# loss, accuracy = model.evaluate(X_test_padded, y_test)
# print(f"Loss: {loss}")
# print(f"Accuracy: {accuracy}")

# model.save('mental_health_model.h5')

# # Muat model dan tokenizer
# model = tf.keras.models.load_model('mental_health_model.h5')
# tokenizer = joblib.load('tokenizer.pkl')
# label_encoder = joblib.load('label_encoder.pkl')

# # Prediksi teks baru
# new_text = ["im bipolar"]
# # Mengubah teks menjadi sequence menggunakan tokenizer yang sudah dilatih sebelumnya
# new_seq = tokenizer.texts_to_sequences(new_text)

# # # Melakukan padding untuk memastikan panjangnya sesuai
# # new_pad = pad_sequences(new_seq, maxlen=100, padding='post')

# # # Menggunakan model untuk memprediksi probabilitas setiap kelas
# # probabilities = model.predict(new_pad)[0]  # Ambil hasil prediksi pertama (karena hanya ada satu input)

# # # Mengambil kelas yang sesuai dengan probabilitas
# # predicted_classes = label_encoder.classes_

# # # Menggabungkan kelas dengan probabilitasnya
# # class_probabilities = dict(zip(predicted_classes, probabilities))

# # # Menampilkan hasil prediksi dalam bentuk persentase
# # for class_name, probability in class_probabilities.items():
# #     print(f"{class_name}: {probability * 100:.2f}%")
# new_seq = tokenizer.texts_to_sequences(new_text)
# new_padded = pad_sequences(new_seq, maxlen=100, padding='post')
# predicted_class = model.predict(new_padded).argmax(axis=-1)

# # # Menghitung persentase prediksi untuk setiap label
# # predictions_percentage = pd.Series(predicted_class).value_counts(normalize=True) * 100

# # # Menampilkan hasil prediksi dalam bentuk persentase
# # print(predictions_percentage)

# # Decode label
# predicted_label = label_encoder.inverse_transform(predicted_class)
# print(f"Predicted Label: {predicted_label}")
