import pandas as pd
from sklearn import svm
import nltk
import seaborn as sns
import numpy as np
from nltk.corpus import PlaintextCorpusReader
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import string
from nltk.corpus.reader import PlaintextCorpusReader
import codecs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report


# Specify the correct encoding (e.g., 'ISO-8859-1')
encoding = 'ISO-8859-1'

# Download NLTK resources
# nltk.download('stopwords')
# nltk.download('punkt')

# The CSV file is located at "C:\\Users\\96654\\OneDrive\\Desktop\\spamraw.csv"
csv_file_path = "C:\\Users\\96654\\OneDrive\\Desktop\\spamraw.csv"

# Read the CSV file into a pandas DataFrame
sms = pd.read_csv(csv_file_path, encoding='utf-8')
# Displaying information about the DataFrame
#sms.info()

# Displaying the first 5 rows of the DataFrame
#print(sms.head(5))

# Changing the data type of the 'type' column to factor
sms['type'] = sms['type'].astype('category')

# Displaying the first few rows of the DataFrame
#print(sms.head())

# Checking for missing values in each column
missing_values = sms.isna().sum()

# Displaying the result
#print(missing_values)

# Computing the proportion of each unique value in the 'type' column
type_proportions = sms['type'].value_counts(normalize=True)

# Displaying the result
#print(type_proportions)

# Displaying the first 5 rows of the 'text' column using slicing
#print(sms['text'][:5])

# Function to perform text cleansing
def clean_text(text):
    # Remove numerical characters
    text = ''.join([char for char in text if not char.isdigit()])
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove English stopwords
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    text = ' '.join([word for word in word_tokens if word.lower() not in stop_words])
    
    # Remove punctuation marks
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Stemming
    stemmer = PorterStemmer()
    words = word_tokenize(text)
    text = ' '.join([stemmer.stem(word) for word in words])
    
    # Strip double white space
    text = ' '.join(text.split())
    
    return text

# Apply text cleansing to the 'text' column in your DataFrame
sms['text'] = sms['text'].apply(clean_text)

# Now, sms['text'] contains the cleaned text
#print(sms['text'][:5])

# Create a CountVectorizer instance
vectorizer = CountVectorizer()

# Fit and transform the text data to get the Document-Term Matrix (DTM)
dtm = vectorizer.fit_transform(sms['text'])

# Convert the DTM to a DataFrame
dtm_df = pd.DataFrame(dtm.toarray(), columns=vectorizer.get_feature_names_out())

# Display the head of the DataFrame
#print(dtm_df.head())

# Display the DTM in a format 
# print("<<DocumentTermMatrix (documents: {}, terms: {})>>".format(dtm_df.shape[0], dtm_df.shape[1]))
# print("Non-/sparse entries: {}/{}".format(dtm.getnnz(), dtm.shape[0] * dtm.shape[1]))
# print("Sparsity           : {:.2%}".format(1 - dtm.getnnz() / (dtm.shape[0] * dtm.shape[1])))
# print("Maximal term length: {}".format(max(len(term) for term in vectorizer.get_feature_names_out())))
# print("Weighting          : term frequency (tf)")
# print("Sample             :")
# print(dtm_df.head())

# Set seed for reproducibility
np.random.seed(123)

# Generate random indices for the train set (80% of data)
index = np.random.choice(dtm.shape[0], size=int(dtm.shape[0] * 0.8), replace=False)

# Split the data into train and test sets
data_train = dtm[index, :]
data_test = dtm[np.setdiff1d(np.arange(dtm.shape[0]), index), :]

# Create label_train and label_test
label_train = sms.loc[index, 'type']
label_test = sms.loc[np.setdiff1d(np.arange(dtm.shape[0]), index), 'type']

# Display label_train and label_test
# print(label_train)
#print(label_test)

# Display proportions of each unique value in label_train
prop_table = label_train.value_counts(normalize=True)
#print(prop_table)

#Further data preproccessing

# Find terms with frequency higher than 20
sms_freq = [term for term, freq in zip(vectorizer.get_feature_names_out(), data_train.sum(axis=0).tolist()[0]) if freq > 20]

# Display the terms
# print(sms_freq)

# Create data_train_filtered with columns that correspond to sms_freq
data_train_filtered = pd.DataFrame(data_train.toarray(), columns=vectorizer.get_feature_names_out())[sms_freq]

# Display the head of data_train_filtered
# print(data_train_filtered.head())

# Make Bernauli Converter
def bernoulli_conv(x):
    # Convert to a NumPy array
    x = np.array(x)
    
    # Apply the Bernoulli conversion
    x = np.where(x > 0, 1, 0)
    
    return x

# Test the function
result = bernoulli_conv([0, 1, 3, 0, 12, 4, 0.3])
#print(result)

# Apply Bernoulli converter to each column of data_train
data_train_bn = np.apply_along_axis(bernoulli_conv, axis=0, arr=data_train.toarray())

# Apply Bernoulli converter to each column of data_test
data_test_bn = np.apply_along_axis(bernoulli_conv, axis=0, arr=data_test.toarray())

################################################################################
# Display a subset of the data_train_bn array (rows 20 to 29, columns 50 to 59)
# subset_data_train_bn = data_train_bn[20:30, 50:60]
# print(subset_data_train_bn)
################################################################################

# Create a DataFrame with specific row and column indices for illustration
data_train_bn_subset = pd.DataFrame(data_train_bn[20:30, 50:60], index=range(20, 30), columns=range(50, 60))

# Display the subset of the data_train_bn DataFrame
# print("##", "{:<7}".format("Terms"))
# print("##", "{:<7}".format("Docs"), end=" ")
# print(" ".join(f'"{col}"' for col in data_train_bn_subset.columns))
# for idx, row in data_train_bn_subset.iterrows():
#     print(f"##   {idx}", " ".join(f'"{val}"' for val in row))



#MODEL FILTERING "Random Forest Model"
# Create a Random Forest model
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)  

# Train the model
model_rf.fit(data_train_bn, label_train)

# Make predictions on the training set (for demonstration purposes)
train_predictions = model_rf.predict(data_train_bn)

# Print training accuracy (for demonstration purposes)
accuracy_train = accuracy_score(label_train, train_predictions)
#print("Training Accuracy:", accuracy_train)

# Evaluate the model using cross-validation (for more robust evaluation)
cv_scores = cross_val_score(model_rf, data_train_bn, label_train, cv=5, scoring='accuracy')
#print("Cross-Validation Scores:", cv_scores)
#print("Mean Cross-Validation Accuracy:", np.mean(cv_scores))

# Assuming data_train_bn is a NumPy array
df_data_train_bn = pd.DataFrame(data_train_bn, columns=range(data_train_bn.shape[1]))

# Display the head of the DataFrame
# print(df_data_train_bn.head())

# Make predictions on the test data
sms_predClass = model_rf.predict(data_test_bn)


# Display the head of the predictions
# print("sms_predClass:")
# print(sms_predClass[:5])  # Displaying the first 5 rows for illustration

#Model Evaluation
# Calculate confusion matrix
conf_matrix = confusion_matrix(label_test, sms_predClass, labels=["ham", "spam"])

# Display the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)


# Calculate and display classification report
class_report = classification_report(label_test, sms_predClass)
print("\nClassification Report:")
print(class_report)

# Create a confusion matrix heatmap
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='BuPu', xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()







