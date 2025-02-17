import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib

# Load the dataset
dataset = pd.read_csv("data/churn_modelling.csv")

# Data Preprocessing
# Handling CustomerId and Surname columns
dataset.drop(['CustomerId', 'Surname'], axis = 1, inplace = True)

# Geography column
# One-hot encoding
geography_dumies = pd.get_dummies(dataset['Geography'], drop_first = True)
dataset = pd.concat([geography_dumies, dataset], axis = 1)
dataset.drop(['Geography'], axis = 1, inplace = True)

# Gender Column
dataset['Gender'] = dataset['Gender'].apply(lambda x: 0 if x == 'Female' else 1)

# Splitting data into features and targets
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Train_Test Split
X_train, X_test, y_train, y_test = (
    train_test_split(X, y, test_size = 0.2, random_state = 42))

# Initialize and train the XGBoost Regressor
model = xgb.XGBClassifier(max_depth = 4, learning_rate = 0.1, n_estimators = 100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
confusion_matrix(y_test, y_pred)

accuracy_score(y_test, y_pred)

# K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = model,
                             X = X,
                             y = y,
                             scoring = 'accuracy',
                             cv = 10)
print("Average Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

# Save the trained model as a .pkl file
joblib.dump(model, 'model/model.pkl')
print("Model saved as 'model.pkl'")





