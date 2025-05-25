# Laporan Proyek Machine Learning - Sinta Ezra Wati Gulo

## Domain Proyek: Kesehatan
### Latar Belakang
Diabetes mellitus merupakan salah satu penyakit tidak menular kronis yang ditandai dengan tingginya kadar gula darah akibat gangguan produksi atau fungsi insulin dalam tubuh. Penyakit ini mencakup berbagai tipe, termasuk diabetes tipe 1, tipe 2, dan gestasional, dengan tipe 2 sebagai yang paling umum. Namun secara keseluruhan, diabetes telah menjadi tantangan kesehatan global yang serius dan terus mengalami peningkatan prevalensi.

International Diabetes Federation (IDF) pada tahun 2021 melaporkan bahwa terdapat sekitar 537 juta orang dewasa hidup dengan diabetes di seluruh dunia, dan angka ini diproyeksikan akan meningkat menjadi 783 juta pada tahun 2045 (Magliano et al., 2021). Di Indonesia, tren serupa juga terlihat seiring dengan perubahan gaya hidup, pola makan, dan tingkat aktivitas fisik masyarakat. Kondisi ini tidak hanya berdampak pada kualitas hidup individu, tetapi juga memberikan beban signifikan terhadap sistem pelayanan kesehatan dan produktivitas ekonomi nasional.

Upaya deteksi dini menjadi sangat krusial untuk mencegah terjadinya komplikasi jangka panjang seperti penyakit jantung, gagal ginjal, kerusakan saraf, dan kehilangan penglihatan. Sayangnya, banyak penderita diabetes yang tidak menyadari kondisinya hingga memasuki tahap lanjut. Di sinilah teknologi, khususnya kecerdasan buatan dan machine learning, memainkan peran penting dalam mengidentifikasi risiko diabetes secara lebih cepat, akurat, dan efisien berbasis data kesehatan individu.

### Referensi
Magliano, D. J., Boyko, E. J., & IDF Diabetes Atlas 10th Edition Scientific Committee. (2021). Global, regional and country-level diabetes prevalence estimates for 2021 and projections for 2045: Results from the International Diabetes Federation Diabetes Atlas, 10th edition. *Diabetes Research and Clinical Practice, 183*, 109119. [https://doi.org/10.1016/j.diabres.2021.109119](https://doi.org/10.1016/j.diabres.2021.109119)

Kavakiotis, I., Tsave, O., Salifoglou, A., Maglaveras, N., Vlahavas, I., & Chouvarda, I. (2017). Machine learning and data mining methods in diabetes research. *Computational and Structural Biotechnology Journal, 15*, 104–116. [https://doi.org/10.1016/j.csbj.2016.12.005](https://doi.org/10.1016/j.csbj.2016.12.005)

Sharma, T., & Shah, M. (2021). A comprehensive review of machine learning techniques on diabetes detection. *Visual Computing for Industry, Biomedicine, and Art, 4*(1), 30. [https://doi.org/10.1186/s42492-021-00097-7](https://doi.org/10.1186/s42492-021-00097-7)

## Business Understanding
### Problem Statements
- Bagaimana memprediksi status seseorang menderita diabetes (positif/negatif) berdasarkan data kesehatan seperti gender, usia, BMI, level HbA1c, kadar glukosa darah, serta riwayat merokok dan penyakit penyerta (seperti hypertension dan heart disease)?
- Algoritma machine learning mana yang memberikan hasil klasifikasi terbaik dalam mendeteksi diabetes berdasarkan data yang tersedia?
- Apa fitur yang paling berpengaruh terhadap kemungkinan seseorang mengidap diabetes?

### Goals
- Mengembangkan sistem prediksi berbasis machine learning yang mampu mengklasifikasikan status diabetes seseorang secara akurat.
- Membandingkan performa beberapa model klasifikasi untuk menentukan algoritma yang paling optimal dalam konteks ini.
- Melakukan eksplorasi terhadap pengaruh berbagai fitur dalam menentukan risiko diabetes.

### Solution statements
- Menggunakan beberapa algoritma klasifikasi untuk membangun model prediksi diabetes, yaitu:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Support Vector Classifier (SVC)
  - Decision Tree Classifier
  - Random Forest Classifier
- Melakukan serangkaian tahapan preprocessing data seperti:
  - Menghapus data duplikat
  - Menangani missing value dan outlier
  - Encoding data kategorikal dan standarisasi fitur numerikal
- Melatih dan mengevaluasi model menggunakan metrik evaluasi klasifikasi:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
- Membandingkan hasil evaluasi antar model untuk memilih model terbaik yang akan dijadikan solusi akhir.
- Melakukan hyperparameter tuning pada model terbaik untuk mengoptimalkan performa berdasarkan metrik evaluasi.

## Data Understanding
Proyek ini menggunakan dataset Diabetes prediction dataset yang dapat diakses melalui Kaggle pada link berikut [Diabetes prediction dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset).

### Variabel-variabel pada Diabetes prediction dataset adalah sebagai berikut:
* `gender`: mengacu pada jenis kelamin biologis seseorang, yang dapat memengaruhi kerentanan terhadap diabetes. Terdapat tiga kategori: laki-laki, perempuan, dan lainnya.
* `age`: faktor penting karena diabetes lebih sering terdiagnosis pada orang dewasa yang lebih tua. Rentang usia dalam dataset ini adalah 0–80 tahun.
* `hypertension`: kondisi medis di mana tekanan darah dalam arteri terus-menerus tinggi. Bernilai 0 atau 1, di mana: 0 berarti tidak memiliki hipertensi dan 1 berarti memiliki hipertensi.
* `heart_disease`:  kondisi medis lain yang berkaitan dengan peningkatan risiko diabetes. Bernilai 0 atau 1, di mana: 0 berarti tidak memiliki penyakit jantung dan 1 berarti memiliki penyakit jantung.
* `smoking_history`: riwayat merokok juga dianggap sebagai faktor risiko diabetes dan dapat memperburuk komplikasi diabetes. Dalam dataset ini, terdapat 6 kategori yaitu not current, former, No Info, current, never, dan ever.
* `bmi `: ukuran lemak tubuh berdasarkan berat dan tinggi badan. Nilai BMI yang tinggi dikaitkan dengan risiko diabetes yang lebih besar Rentang BMI dalam dataset adalah 10.16 – 71.55. Kategori BMI pada dataset ini yaitu <18.5 = berat badan kurang, 18.5–24.9 = normal, 25–29.9 = kelebihan berat badan, dan ≥30 = obesitas.
* `HbA1c_level`: HbA1c (Hemoglobin A1c) mengukur rata-rata kadar gula darah selama 2–3 bulan terakhir. Nilai yang lebih tinggi menunjukkan risiko lebih besar terkena diabetes.
Umumnya, HbA1c > 6.5% menunjukkan diabetes.
* `blood_glucose_level`:  mengacu pada jumlah gula dalam aliran darah pada satu waktu. Kadar glukosa darah yang tinggi merupakan indikator utama diabetes.
* `diabetes`: variabel target yang diprediksi dalam dataset, dengan 1 menunjukkan menderita diabetes dan 0 menunjukkan tidak menderita diabetes.

### Exploratory Data Analysis 





