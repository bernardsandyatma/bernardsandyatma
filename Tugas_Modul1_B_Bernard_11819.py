import pandas as pd
import numpy as np
 
#load data
 diabetes_csv = pd.read_csv(r'D:\dataset\diabetes.csv')#Disesuaikan dengan tempat penyimpanan file csv Dataset diabetes
 
#load dataset ke dalam dataframe
 df_diabetes = pd.DataFrame(data = diabetes_csv, index = None)
 
 #mengecek data kosong, null, dan nan
 print("data null\n", df_diabetes.isnull().sum())
 print("\n")
 print("data kosong \n", df_diabetes.empty)
 print("\n")
 print("data nan \n", df_diabetes.isna().sum())

 df_diabetes.describe()

 #memuat library untuk train-test split dataset
 from sklearn.model_selection import train_test_split
 #memuat nilai fitur dalam variabel X, drop Outcome
 #axis = 1 digunakan untuk menghapus kolom
 X = df_diabetes.drop(columns=['Outcome'], axis = 1)
 #memuat nilai label dalam variabel y
 y = df_diabetes['Outcome']
 #membuat variabel X_train, X_test, y_train, dan y_test untuk menampung hasil split
 ##nilai random state diganti dengan 2 digit npm terakhir
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
 print("bentuk X_train", X_train.shape)
 print("bentuk X_test", X_test.shape)
 print("bentuk y_train", y_train.shape)
 print("bentuk y_test", y_test.shape)
 print("y_train \n", y_train)
 print("y_test \n", y_test)

 #import library untuk model machine learning
 from sklearn.svm import SVC
 from sklearn.ensemble import RandomForestClassifier
 #membuat obyek model machine learning dan setting parameternya
 ##nilai random state diganti dengan 2 digit npm terakhir
 SVM = SVC(C = 1, gamma= 0.01, random_state=42) 
RF = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=42)
 #fungsi fit digunakan untuk melatih machine learning
 SVM.fit(X_train, y_train)
 RF.fit(X_train, y_train)

 #membuat array untuk X baru yang akan diprediksi
 X_new = np.array([[3, 197, 30, 19, 0, 44.8, 0.567, 55]])
 print("X_new yang akan diprediksi", X_new.shape)
 #prediksi label dari X baru
 svm_predict = SVM.predict(X_new)
 print("Label prediksi SVM", svm_predict)
 rf_predict = RF.predict(X_new)
 print("Label prediksi RF", rf_predict)

 #menggunakan fungsi predict untuk memprediksi label X_test
 y_pred_svm = SVM.predict(X_test)
 y_pred_rf = RF.predict(X_test)
 print("Hasil prediksi SVM pada X_test:", y_pred_svm)
 print("Hasil prediksi RF pada X_test:", y_pred_rf)

 #menggunakan fungsi score untuk mengukur akurasi prediksi model
 print("Akurasi model SVM:", round(SVM.score(X_test, y_test), 3))
 print("Akurasi model RF:", round(RF.score(X_test, y_test), 3))

 
 #simpan model menggunakan library Pickle
 import pickle
 with open('rf_diabetes_model.pkl', 'wb') as f:
 pickle.dump((RF), f)
 ##File pickle(.pkl) akan tersimpan di folder yang sama dengan file notebook
 print("Model RF berhasil disimpan")