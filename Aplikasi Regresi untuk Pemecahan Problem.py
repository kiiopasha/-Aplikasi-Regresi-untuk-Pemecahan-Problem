import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Dataset Durasi Waktu Belajar (TB) dan Nilai Ujian
data = {
    'TB': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Durasi Waktu Belajar (jam)
    'Nilai': [52, 55, 60, 63, 65, 70, 72, 75, 80, 85]  # Nilai Ujian
}

df = pd.DataFrame(data)

# Membagi data menjadi variabel independen (X) dan dependen (y)
X = df[['TB']].values
y = df['Nilai'].values

# Membagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 1. Model Linear
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

print("Model Linear")
print(f"Mean Squared Error: {mse_linear}")
print(f"R^2 Score: {r2_linear}")

# 2. Model Eksponensial
# Transformasi logaritmik pada y
log_y_train = np.log(y_train)
exp_model = LinearRegression()
exp_model.fit(X_train, log_y_train)
log_y_pred_exp = exp_model.predict(X_test)
y_pred_exp = np.exp(log_y_pred_exp)

mse_exp = mean_squared_error(y_test, y_pred_exp)
r2_exp = r2_score(y_test, y_pred_exp)

print("\nModel Eksponensial")
print(f"Mean Squared Error: {mse_exp}")
print(f"R^2 Score: {r2_exp}")

# Visualisasi hasil
plt.figure(figsize=(14, 6))

# Plot Model Linear
plt.subplot(1, 2, 1)
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred_linear, color='red', linewidth=2, label='Predicted')
plt.xlabel('Durasi Waktu Belajar (TB)')
plt.ylabel('Nilai Ujian')
plt.title('Linear Regression')
plt.legend()

# Plot Model Eksponensial
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred_exp, color='red', linewidth=2, label='Predicted')
plt.xlabel('Durasi Waktu Belajar (TB)')
plt.ylabel('Nilai Ujian')
plt.title('Exponential Regression')
plt.legend()

plt.show()
