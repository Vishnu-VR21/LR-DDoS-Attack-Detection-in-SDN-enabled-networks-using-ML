Python 3.12.2 (tags/v3.12.2:6abddd9, Feb  6 2024, 21:26:36) [MSC v.1937 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import os

import logging

import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder


from sklearn.model_selection import train_test_split

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S", level=logging.INFO)

pd.set_option("display.max_rows", 85)

DIR_PATH = "C:/Program Files/Python312"
PROCESSED_DIR_PATH = "D:/"

FILE_PATH = os.path.join(DIR_PATH, "modified_file.csv")
def _label_encoding() -> LabelEncoder:
    le = LabelEncoder()

    labels = pd.read_csv(FILE_PATH, usecols=['Label'], skipinitialspace=True)

SyntaxError: unexpected indent

def _label_encoding() -> LabelEncoder:
    # Create Label Encoder
    le = LabelEncoder()

    # Read Label column from all dataset files
    labels = pd.read_csv(FILE_PATH, usecols=['Label'], skipinitialspace=True)

    # Fit the labels data to Label Encoder
    le.fit(labels.Label)

    # Saving the label encoder
    np.save(os.path.join(PROCESSED_DIR_PATH, 'label_encoder.npy'), le.classes_)

    # Log the result.
    logging.info("Total rows: {}".format(labels.shape))
    logging.info("Class distribution:\n{}\n".format(labels.Label.value_counts()))

    return le

def _process(df: pd.DataFrame, le: LabelEncoder) -> (np.ndarray, np.ndarray):
    # Label encoding
    df.Label = le.transform(df.Label)

    # Fill NaN with average value of each class in this dataset
    nan_rows = df[df.isna().any(axis=1)].shape[0]
    logging.info("Fill NaN in {} rows with average value of each class.".format(nan_rows))
    df.iloc[:, df.columns != "Label"] = df.groupby("Label").transform(lambda x: x.fillna(x.mean()))

    # Change inf value with maximum value of each class
    inf_rows = df[df.isin([np.inf]).any(axis=1)].shape[0]
    logging.info("Replace Inf in {} rows with maximum value of each class.".format(inf_rows))
    # Temporary replace inf with NaN
    df = df.replace([np.inf], np.nan)
    # Replace inf with maximum value of each class in this dataset
    df.iloc[:, df.columns != "Label"] = df.groupby("Label").transform(lambda x: x.fillna(x.max()))

    # Change negative value with minimum positive value of each class
    logging.info("Replace negative values with minimum value of each class.")
    # Temporary replace negative value with NaN
    df[df < 0] = np.nan
    # Replace negative value with minimum value of each class in this dataset
    df.iloc[:, df.columns != "Label"] = df.groupby("Label").transform(lambda x: x.fillna(x.min()))

    return df

def _split_train_test(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    # Sampling the dataset
    x = df.iloc[:, df.columns != 'Label']
    y = df[['Label']]

    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.20,
                                                        random_state=np.random.randint(10))

    del x, y

    train = pd.concat([x_train, y_train], axis=1, sort=False)
    test = pd.concat([x_test, y_test], axis=1, sort=False)

    return train, test

def _to_csv(df: pd.DataFrame, saving_path: str):
    # if file does not exist write header
    if not os.path.isfile(saving_path):
        df.to_csv(saving_path, index=False)
    # else it exists so append without writing the header
    else:
        df.to_csv(saving_path, index=False, mode='a', header=False)

def _preprocessing_all(le: LabelEncoder, chunksize=1000000):
    # Preprocess all file
    for chunk in pd.read_csv(FILE_PATH, skipinitialspace=True, chunksize=chunksize):
        train, test = _split_train_test(_process(chunk, le))
        _to_csv(train, os.path.join(PROCESSED_DIR_PATH, "train_MachineLearningCVE.csv"))
        _to_csv(test, os.path.join(PROCESSED_DIR_PATH, "test_MachineLearningCVE.csv"))

        
_label_enc = _label_encoding()
_preprocessing_all(_label_enc)
SyntaxError: multiple statements found while compiling a single statement
_label_enc = _label_encoding()
21:24:47 INFO Total rows: (1042557, 1)
21:24:47 INFO Class distribution:
Label
BENIGN      629074
DoS         194642
DDoS        128022
PortScan     90819
Name: count, dtype: int64



_preprocessing_all(_label_enc)

21:25:02 INFO Fill NaN in 0 rows with average value of each class.
21:25:06 INFO Replace Inf in 629 rows with maximum value of each class.
21:25:10 INFO Replace negative values with minimum value of each class.
21:25:52 INFO Fill NaN in 0 rows with average value of each class.
21:25:52 INFO Replace Inf in 29 rows with maximum value of each class.
21:25:52 INFO Replace negative values with minimum value of each class.


def _chi_square_feature_selection(df: pd.DataFrame, le: LabelEncoder, k: int) -> pd.DataFrame:
    # Encode the target variable
    df['Label'] = le.transform(df['Label'])

    # Separate features and target
    X = df.drop(columns=['Label'])
    y = df['Label']

    # Apply chi-square feature selection
    selector = SelectKBest(chi2, k=k)
    X_selected = selector.fit_transform(X, y)

    # Get selected feature indices
    selected_indices = selector.get_support(indices=True)

    # Get selected feature names
    selected_features = X.columns[selected_indices]

    # Combine selected features and target into DataFrame
    selected_df = pd.DataFrame(data=X_selected, columns=selected_features)
    selected_df['Label'] = y.values

    return selected_df

_label_enc = _label_encoding()

21:27:01 INFO Total rows: (1042557, 1)
21:27:01 INFO Class distribution:
Label
BENIGN      629074
DoS         194642
DDoS        128022
PortScan     90819
Name: count, dtype: int64


_preprocessing_all(_label_enc)

21:27:15 INFO Fill NaN in 0 rows with average value of each class.
21:27:19 INFO Replace Inf in 629 rows with maximum value of each class.
21:27:23 INFO Replace negative values with minimum value of each class.
21:28:03 INFO Fill NaN in 0 rows with average value of each class.
21:28:04 INFO Replace Inf in 29 rows with maximum value of each class.
21:28:04 INFO Replace negative values with minimum value of each class.


train_data = pd.read_csv(os.path.join(PROCESSED_DIR_PATH, "train_MachineLearningCVE.csv"))



selected_train_data = _chi_square_feature_selection(train_data, _label_enc, k=10)

Traceback (most recent call last):
  File "C:\Users\abhi'\AppData\Roaming\Python\Python312\site-packages\sklearn\utils\_encode.py", line 225, in _encode
    return _map_to_integer(values, uniques)
  File "C:\Users\abhi'\AppData\Roaming\Python\Python312\site-packages\sklearn\utils\_encode.py", line 165, in _map_to_integer
    return np.array([table[v] for v in values])
  File "C:\Users\abhi'\AppData\Roaming\Python\Python312\site-packages\sklearn\utils\_encode.py", line 159, in __missing__
    raise KeyError(key)
KeyError: 0

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<pyshell#30>", line 1, in <module>
    selected_train_data = _chi_square_feature_selection(train_data, _label_enc, k=10)
  File "<pyshell#25>", line 3, in _chi_square_feature_selection
    df['Label'] = le.transform(df['Label'])
  File "C:\Users\abhi'\AppData\Roaming\Python\Python312\site-packages\sklearn\preprocessing\_label.py", line 137, in transform
    return _encode(y, uniques=self.classes_)
  File "C:\Users\abhi'\AppData\Roaming\Python\Python312\site-packages\sklearn\utils\_encode.py", line 227, in _encode
    raise ValueError(f"y contains previously unseen labels: {str(e)}")
ValueError: y contains previously unseen labels: 0
def _chi_square_feature_selection(df: pd.DataFrame, le: LabelEncoder, k: int) -> pd.DataFrame:
    # Encode the target variable, ignoring unseen labels
    df['Label'] = df['Label'].apply(lambda x: le.transform([x])[0] if x in le.classes_ else np.nan)

    # Drop rows with NaN labels
    df.dropna(subset=['Label'], inplace=True)

    # Separate features and target
    X = df.drop(columns=['Label'])
    y = df['Label']

    # Apply chi-square feature selection
    selector = SelectKBest(chi2, k=k)
    X_selected = selector.fit_transform(X, y)

    # Get selected feature indices
    selected_indices = selector.get_support(indices=True)

    # Get selected feature names
    selected_features = X.columns[selected_indices]

    # Combine selected features and target into DataFrame
    selected_df = pd.DataFrame(data=X_selected, columns=selected_features)
    selected_df['Label'] = y.values

    return selected_df

_label_enc = _label_encoding()

21:29:48 INFO Total rows: (1042557, 1)
21:29:48 INFO Class distribution:
Label
BENIGN      629074
DoS         194642
DDoS        128022
PortScan     90819
Name: count, dtype: int64


_preprocessing_all(_label_enc)

21:29:59 INFO Fill NaN in 0 rows with average value of each class.
21:30:02 INFO Replace Inf in 629 rows with maximum value of each class.
21:30:06 INFO Replace negative values with minimum value of each class.
21:30:47 INFO Fill NaN in 0 rows with average value of each class.
21:30:47 INFO Replace Inf in 29 rows with maximum value of each class.
21:30:47 INFO Replace negative values with minimum value of each class.

selected_train_data = _chi_square_feature_selection(train_data, _label_enc, k=10)

Traceback (most recent call last):
  File "<pyshell#34>", line 1, in <module>
    selected_train_data = _chi_square_feature_selection(train_data, _label_enc, k=10)
  File "<pyshell#31>", line 13, in _chi_square_feature_selection
    selector = SelectKBest(chi2, k=k)
NameError: name 'SelectKBest' is not defined

train_data = pd.read_csv(os.path.join(PROCESSED_DIR_PATH, "train_MachineLearningCVE.csv"))


selected_train_data = _chi_square_feature_selection(train_data, _label_enc, k=10)

Traceback (most recent call last):
  File "<pyshell#36>", line 1, in <module>
    selected_train_data = _chi_square_feature_selection(train_data, _label_enc, k=10)
  File "<pyshell#31>", line 13, in _chi_square_feature_selection
    selector = SelectKBest(chi2, k=k)
NameError: name 'SelectKBest' is not defined


from sklearn.feature_selection import SelectKBest, chi2

def _chi_square_feature_selection(df: pd.DataFrame, le: LabelEncoder, k: int) -> pd.DataFrame:
    # Encode the target variable, ignoring unseen labels
    df['Label'] = df['Label'].apply(lambda x: le.transform([x])[0] if x in le.classes_ else np.nan)

    # Drop rows with NaN labels
    df.dropna(subset=['Label'], inplace=True)

    # Separate features and target
    X = df.drop(columns=['Label'])
    y = df['Label']

    # Apply chi-square feature selection
    selector = SelectKBest(chi2, k=k)
    X_selected = selector.fit_transform(X, y)

    # Get selected feature indices
    selected_indices = selector.get_support(indices=True)

    # Get selected feature names
    selected_features = X.columns[selected_indices]

    # Combine selected features and target into DataFrame
    selected_df = pd.DataFrame(data=X_selected, columns=selected_features)
    selected_df['Label'] = y.values

    return selected_df

SyntaxError: multiple statements found while compiling a single statement
from sklearn.feature_selection import SelectKBest, chi2

def _chi_square_feature_selection(df: pd.DataFrame, le: LabelEncoder, k: int) -> pd.DataFrame:
    # Encode the target variable, ignoring unseen labels
    df['Label'] = df['Label'].apply(lambda x: le.transform([x])[0] if x in le.classes_ else np.nan)

    # Drop rows with NaN labels
    df.dropna(subset=['Label'], inplace=True)

    # Separate features and target
    X = df.drop(columns=['Label'])
    y = df['Label']

    # Apply chi-square feature selection
    selector = SelectKBest(chi2, k=k)
    X_selected = selector.fit_transform(X, y)

    # Get selected feature indices
    selected_indices = selector.get_support(indices=True)

    # Get selected feature names
    selected_features = X.columns[selected_indices]

    # Combine selected features and target into DataFrame
    selected_df = pd.DataFrame(data=X_selected, columns=selected_features)
    selected_df['Label'] = y.values

    return selected_df

train_data = pd.read_csv(os.path.join(PROCESSED_DIR_PATH, "train_MachineLearningCVE.csv"))

selected_train_data = _chi_square_feature_selection(train_data, _label_enc, k=10)

Traceback (most recent call last):
  File "<pyshell#42>", line 1, in <module>
    selected_train_data = _chi_square_feature_selection(train_data, _label_enc, k=10)
  File "<pyshell#40>", line 14, in _chi_square_feature_selection
    X_selected = selector.fit_transform(X, y)
  File "C:\Users\abhi'\AppData\Roaming\Python\Python312\site-packages\sklearn\utils\_set_output.py", line 295, in wrapped
    data_to_wrap = f(self, X, *args, **kwargs)
  File "C:\Users\abhi'\AppData\Roaming\Python\Python312\site-packages\sklearn\base.py", line 1101, in fit_transform
    return self.fit(X, y, **fit_params).transform(X)
  File "C:\Users\abhi'\AppData\Roaming\Python\Python312\site-packages\sklearn\base.py", line 1474, in wrapper
    return fit_method(estimator, *args, **kwargs)
  File "C:\Users\abhi'\AppData\Roaming\Python\Python312\site-packages\sklearn\feature_selection\_univariate_selection.py", line 562, in fit
    X, y = self._validate_data(
  File "C:\Users\abhi'\AppData\Roaming\Python\Python312\site-packages\sklearn\base.py", line 650, in _validate_data
    X, y = check_X_y(X, y, **check_params)
  File "C:\Users\abhi'\AppData\Roaming\Python\Python312\site-packages\sklearn\utils\validation.py", line 1263, in check_X_y
    X = check_array(
  File "C:\Users\abhi'\AppData\Roaming\Python\Python312\site-packages\sklearn\utils\validation.py", line 1072, in check_array
    raise ValueError(
ValueError: Found array with 0 sample(s) (shape=(0, 79)) while a minimum of 1 is required by SelectKBest.
from sklearn.feature_selection import SelectKBest, chi2

# Existing code for preprocessing steps (omitted for brevity)

def _chi_square_feature_selection(train_df: pd.DataFrame, test_df: pd.DataFrame, k: int) -> (pd.DataFrame, pd.DataFrame):
    # Separate features and target in training data
    X_train = train_df.drop(columns=['Label'])
    y_train = train_df['Label']

    # Apply chi-square feature selection on training data
    selector = SelectKBest(chi2, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)

    # Get selected feature indices
    selected_indices = selector.get_support(indices=True)

    # Get selected feature names
    selected_features = X_train.columns[selected_indices]

    # Filter features in testing data
    X_test_selected = test_df[selected_features]

    return X_train_selected, X_test_selected
SyntaxError: multiple statements found while compiling a single statement
from sklearn.feature_selection import SelectKBest, chi2



def _chi_square_feature_selection(train_df: pd.DataFrame, test_df: pd.DataFrame, k: int) -> (pd.DataFrame, pd.DataFrame):
    # Separate features and target in training data
...     X_train = train_df.drop(columns=['Label'])
...     y_train = train_df['Label']
... 
...     # Apply chi-square feature selection on training data
...     selector = SelectKBest(chi2, k=k)
...     X_train_selected = selector.fit_transform(X_train, y_train)
... 
...     # Get selected feature indices
...     selected_indices = selector.get_support(indices=True)
... 
...     # Get selected feature names
...     selected_features = X_train.columns[selected_indices]
... 
...     # Filter features in testing data
...     X_test_selected = test_df[selected_features]
... 
...     return X_train_selected, X_test_selected
... 
>>> train_data = pd.read_csv(os.path.join(PROCESSED_DIR_PATH, "train_MachineLearningCVE.csv"))
... 
>>> test_data = pd.read_csv(os.path.join(PROCESSED_DIR_PATH, "test_MachineLearningCVE.csv"))
... 
>>> X_train_selected, X_test_selected = _chi_square_feature_selection(train_data, test_data, k=10)
... 
>>> selected_train_data = pd.concat([pd.DataFrame(X_train_selected), train_data['Label']], axis=1)
... 
>>> selected_test_data = pd.concat([pd.DataFrame(X_test_selected), test_data['Label']], axis=1)
... 
>>> selected_train_data.to_csv(os.path.join(PROCESSED_DIR_PATH, "selected_train_MachineLearningCVE.csv"), index=False)
... 
>>> selected_test_data.to_csv(os.path.join(PROCESSED_DIR_PATH, "selected_test_MachineLearningCVE.csv"), index=False)
... 
