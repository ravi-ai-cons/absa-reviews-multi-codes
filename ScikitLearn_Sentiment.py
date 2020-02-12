from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import random

data = []
data_labels = []
with open("./pos_tweets.txt") as f:
    for i in f:
        data.append(i)
        data_labels.append('pos')

with open("./neg_tweets.txt") as f:
    for i in f:
        data.append(i)
        data_labels.append('neg')

vectorizer = CountVectorizer(analyzer='word', lowercase=False)
features = vectorizer.fit_transform(data)
features_nd = features.toarray()

X_train, X_test, y_train, y_test = train_test_split(features_nd, data_labels, train_size=0.8, random_state=12)
logit_model = LogisticRegression()
logit_model = logit_model.fit(X=X_train, y=y_train)
y_pred = logit_model.predict(X_test)

# j = random.randint(0, len(X_test)-2)
for i in range(0, len(X_test)):
    print(y_pred[0])
    ind = features_nd.tolist().index(X_test[i].tolist())
    print(data[ind].strip())

print(accuracy_score(y_test, y_pred))
