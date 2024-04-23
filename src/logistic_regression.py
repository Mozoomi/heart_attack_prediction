import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap


heart_data = pd.read_csv('data/clean_data.csv')

logistic_model = None

X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

#TRAIN MODEL
#create models
logistic_model = LogisticRegression()
#train the models
logistic_model.fit(X_train, Y_train)
   
#EVALUATE MODEL
X_train_prediction = logistic_model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("TRAIN ACCURACY: ", training_data_accuracy)
X_test_prediction = logistic_model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("TEST ACCURACY: ", training_data_accuracy)

#Prediction System
def prediction():
   input_data = (37,1,2,130,250,0,1,187,0,3.5,0,0,2,1)
   data_array = np.asarray(input_data)
   reshaped_data = data_array.reshape(1, -1)
   prediction = logistic_model.predict(reshaped_data)
   print(prediction)

colors = [(0, '#117733'),     # top left: darker green
          (0.25, '#6611AA'),  # bottom left: darker purple
          (0.75, '#6611AA'),  # top right: darker purple
          (1, '#FFDD44')]    # bottom right: darker yellow

custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)


conf_matrix = confusion_matrix(Y_test, X_test_prediction)

# Plot confusion matrix
class_labels = ['No Heart Attack', 'Heart Attack']
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap=custom_cmap, fmt='d', cbar=False, xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()