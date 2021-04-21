# Task 1 : Prediction using Supervised ML

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Collect data
url = "http://bit.ly/w-data"
data = pd.read_csv(url)
print(data)

# Prepare Data
Hours = data['Hours'].values
Scores = data['Scores'].values
print(Scores)
print(Hours)

# Plot Data
plt.scatter(Hours, Scores)
plt.ylim(0, 100)
plt.title('Student Chart')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()

# Prepare test data
time_train, time_test, score_train, score_test = train_test_split(Hours, Scores, test_size=0.2, random_state=0)
print(time_train)
print(time_test)
print(score_train)
print(score_test)

# Model
model = LinearRegression()
model.fit(time_train.reshape(-1, 1), score_train.reshape(-1, 1))
print("Training complete.")

# plot output line
line = model.coef_ * Hours.reshape(-1, 1) + model.intercept_
plt.scatter(Hours, Scores)
plt.plot(Hours, line, color='r')
plt.ylim(0, 100)
plt.title('Student Chart')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()

# Make Prediction on Test Data (Accuracy)
model_predict = model.predict(time_test.reshape(-1, 1))
array = np.vstack((score_test.reshape(1, -1), model_predict.reshape(1, -1))).T
df = pd.DataFrame(array, columns=['Original', 'Predict'])
print(df)

plt.plot(df['Original'], color='r')
plt.plot(df['Predict'])
plt.ylim(0, 100)
plt.title('Student Chart')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()

# Predict data
input_time = 9.25
pred_score = model.predict(np.array(input_time).reshape(-1,1))
print(f"Time: {input_time}")
print(f"Prediction: {pred_score[0]}")
