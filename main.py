import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Import data set for training
df = pd.read_csv('census-income.data.csv')

#Change the labels in the dataframe
labels = ["age"," workclass"," fnlwgt"," education"," education-num"," marital status"," occupation"," relationship"," race"," sex"," capital-gain"," capital-loss"," hours-per-week"," native country"," income"]
for x in range(len(df.columns)):
    df.columns.values[x] = labels[x]

# For any column that has a "?", drop that column by setting the value as NaN and dropping it
df[" workclass"] = df[" workclass"].replace(" ?", np.nan)
df[" occupation"] = df[" occupation"].replace(" ?", np.nan)
df[" native country"] = df[" native country"].replace(" ?", np.nan)
df.dropna(how='any', inplace=True)

# Convert the two possible incomes into 0 and 1 to allow for correlation test
df[" income"] = df[" income"].replace(" <=50K", 0)
df[" income"] = df[" income"].replace(" >50K", 1)

# Check what values in the training set have a correlation to the income which is greater than 15% (including inverse
# correlation)
for column in df:
    if 0.15 < df[column].astype('category').cat.codes.corr(df[' income']) or df[column].astype(
            'category').cat.codes.corr(df[' income']) < -0.15:
        df[column] = df[column].astype('category').cat.codes
    else:
        df.drop([column], axis=1, inplace=True)

# Allocate the training data to x_train and y_train
df_x = df[['age', ' education-num', ' marital status', ' relationship', ' sex', ' capital-gain', ' capital-loss',
              ' hours-per-week']]
df_y = df[' income']

X_train, train_X_test, y_train, train_y_test = train_test_split(df_x, df_y, test_size=0.33)

# Create model and fit it using the training data
model = LogisticRegression(max_iter=100000)
model.fit(X_train, y_train)

# Import the test data
test_df = pd.read_csv('census-income.test.csv')

#Change the labels in the dataframe
labels = ["age"," workclass"," fnlwgt"," education"," education-num"," marital status"," occupation"," relationship"," race"," sex"," capital-gain"," capital-loss"," hours-per-week"," native country"," income"]
for x in range(len(test_df.columns)):
    test_df.columns.values[x] = labels[x]

# Format the test data
for column in ['age', ' education-num', ' marital status', ' relationship', ' sex', ' capital-gain', ' capital-loss',
               ' hours-per-week']:
    test_df[column] = test_df[column].astype('category').cat.codes
test_df[" income"] = test_df[" income"].replace(" <=50K.", 0)
test_df[" income"] = test_df[" income"].replace(" >50K.", 1)

# Allocate the test data
x_test = test_df[['age', ' education-num', ' marital status', ' relationship', ' sex', ' capital-gain', ' capital-loss',
                  ' hours-per-week']]
y_test = test_df[' income']

# Run model and obtain predictions on the test set
test_predictions = model.predict(x_test)

# Run model and obtain predictions from train set
train_predictions = model.predict(train_X_test)

# Test accuracy of testing data predictions
accuracy = accuracy_score(y_test, test_predictions)

# Test accuracy of training data predictions
train_accuracy = accuracy_score(train_y_test, train_predictions)

# Print accuracy of model
print(f"Testing data set prediction is: {accuracy}")
print(f"Training data set prediction accuracy is: {train_accuracy}")
