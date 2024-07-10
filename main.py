import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
import seaborn as sns

# Function to detect spam
def detect_spam(message, model, vectorizer):
    message_tfidf = vectorizer.transform([message])
    prediction = model.predict(message_tfidf)
    return 'Spam' if prediction[0] == 1 else 'Ham'

# Load dataset
df = pd.read_csv('spam.csv', encoding='latin-1')

# Drop unnecessary columns
df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])

# Rename columns
df.columns = ['label', 'message']

# Map labels to binary values
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Visualize the distribution of spam and ham messages
label_counts = df['label'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(label_counts, labels=['Ham', 'Spam'], autopct='%1.1f%%', colors=['skyblue', 'salmon'])
plt.title('Distribution of Spam and Ham Messages')
plt.show()

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.3, random_state=42)

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Transform test data
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialize Logistic Regression model
logreg = LogisticRegression()

# Train the model
logreg.fit(X_train_tfidf, y_train)

# Predict on the test data
y_pred = logreg.predict(X_test_tfidf)
y_pred_prob = logreg.predict_proba(X_test_tfidf)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Test the spam detection function with multiple messages
test_messages = [
    "Congratulations! You've won a free ticket to the Bahamas. Call now!",
    "Reminder: Your appointment is scheduled for tomorrow at 10 AM.",
    "URGENT! Your account has been compromised. Please contact support immediately.",
    "Hey, are we still meeting for lunch tomorrow?",
    "You have received a bonus of $500. Click here to claim your prize."
]

for i, msg in enumerate(test_messages, 1):
    prediction = detect_spam(msg, logreg, tfidf_vectorizer)
    print(f"Message {i}: {msg}\nPrediction: {prediction}\n")
