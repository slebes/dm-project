import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import TruncatedSVD


# Generate the df
df = pd.read_csv('abstractdata5.csv', sep='#', names=['id', 'class', 'title', 'abstract'])

# Remove punctuation
df['joined'] = df['title'] + df['abstract']
df['noPunct'] = df['joined'].str.replace('[^\w\s]','')

# Remove digits
df['noPunct'] = df['noPunct'].str.replace('\d+', '')

# Remove unicode
df['noPunct'].str.encode('ascii', 'ignore').str.decode('ascii')

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
v = TfidfVectorizer(max_df=0.25, min_df=1, ngram_range=(1,2), stop_words=['data','use', 'propos'])
x = v.fit_transform(df['preprocessed'])


# LSA
lsaApplied = TruncatedSVD(n_components=100, random_state=5).fit_transform(x)

# Spectral clustering with cosine distance. Laplacian ?
clustering = SpectralClustering(n_clusters=5, affinity='cosine', random_state=2, assign_labels='discretize', ).fit(lsaApplied)

labels = clustering.labels_

nmi = normalized_mutual_info_score(labels, df['class'], )
print('\nNMI value: ', nmi)

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
  "\n30 most common words in each cluster. Note that stopwords are contained and words are stemmed.\n"
  "\ncluster0:\n", c0, "\n------------------------------------------\n",
  "cluster1:\n", c1, "\n------------------------------------------\n",
  "cluster2:\n", c2, "\n------------------------------------------\n",
  "cluster3:\n", c3, "\n------------------------------------------\n",
  "cluster4:\n", c4, "\n------------------------------------------\n")