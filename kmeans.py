import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics.cluster import normalized_mutual_info_score

# Generate the df
df = pd.read_csv('abstractdata5.csv', sep='#', names=['id', 'class', 'title', 'abstract'])

# Remove punctuation
df['joined'] = df['title'] + df['abstract']
df['noPunct'] = df['joined'].str.replace('[^\w\s]','')

# Lowercase
df['noPunct'] = df['noPunct'].str.lower()

# Remove stopwords
stop = stopwords.words('english')
df['noStopwords'] = df['noPunct'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# Stem
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()

def stem_text(text):
    ps = SnowballStemmer('english')
    return " ".join([ps.stem(w) for w in w_tokenizer.tokenize(text)])

df['preprocessed'] = df['noStopwords'].apply(stem_text)

# tf-idf
v = TfidfVectorizer()
x = v.fit_transform(df['preprocessed'])

# Calculate NMI
clustering = KMeans(n_clusters=5, n_init=10, random_state=2).fit(x)

labels = clustering.labels_

nmi = normalized_mutual_info_score(labels, df['class'])

print(nmi)

df["clusteringLabel"] = labels

def countAndSort(cluster):
  v = CountVectorizer()
  x = v.fit_transform(cluster)

  counts = x.toarray().sum(axis=0)
  features = v.get_feature_names_out()

  d = dict(zip(features, counts))
  sortedByCount = sorted(d.items(), key=lambda x: x[1], reverse=True)
  
  return sortedByCount

cluster0 = df[df['clusteringLabel'] == 0]
cluster1 = df[df['clusteringLabel'] == 1]
cluster2 = df[df['clusteringLabel'] == 2]
cluster3 = df[df['clusteringLabel'] == 3]
cluster4 = df[df['clusteringLabel'] == 4]

c0 = countAndSort(cluster0['preprocessed'])[0:30]
c1 = countAndSort(cluster1['preprocessed'])[0:30]
c2 = countAndSort(cluster2['preprocessed'])[0:30]
c3 = countAndSort(cluster3['preprocessed'])[0:30]
c4 = countAndSort(cluster4['preprocessed'])[0:30]

print(
  "\ncluster0:\n", c0, "\n------------------------------------------\n",
  "cluster1:\n", c1, "\n------------------------------------------\n",
  "cluster2:\n", c2, "\n------------------------------------------\n",
  "cluster3:\n", c3, "\n------------------------------------------\n",
  "cluster4:\n", c4, "\n------------------------------------------\n")
