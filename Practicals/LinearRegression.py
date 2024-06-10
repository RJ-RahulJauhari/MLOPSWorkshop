import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


file_path = './Datasets/wine-quality.csv'
data = pd.read_csv(file_path)

print(data.isnull().sum())

X = data.drop(columns='quality')
y = data['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R²): {r2}')

# Actual vs Predicted plot
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Quality')
plt.ylabel('Predicted Quality')
plt.title('Actual vs Predicted Wine Quality')

# Residuals plot
plt.subplot(1, 2, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Quality')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Wine Quality')

plt.tight_layout()
plt.show()

x_feature = 'residual sugar'
y_feature = 'alcohol'
hue_feature = 'quality'

plt.figure(figsize=(12, 8))
scatter = sns.scatterplot(data=data, x=x_feature, y=y_feature, hue=hue_feature, palette='viridis', alpha=0.7)

plt.title('Residual Sugar vs Alcohol Content by Wine Quality')
plt.xlabel('Residual Sugar')
plt.ylabel('Alcohol Content')

plt.legend(title='Quality', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
