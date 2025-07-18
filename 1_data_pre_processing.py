import nltk
import spacy
import pandas as pd
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from tqdm import tqdm
from models.consts import MAX_VOCAB_SIZE, UNK_TOKEN
import pickle
tqdm.pandas()

# download packages for pre-processing
nltk.download('punkt')
nltk.download('stopwords')
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")
nltk.download('punkt_tab')
STOPWORDS = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer('english')
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Task 1: Dataset Acquisition and Pre-Processing
# data importing & initial filtering by "business"/"sports" NewsType
news_type = "sports"
file_path = "Data/Articles.csv"
df = pd.read_csv(file_path, encoding='ISO-8859-1')
df = df[df['NewsType'] == news_type]
df = df.dropna(subset=['Article'])


def clean_text(text):
    # https://www.geeksforgeeks.org/machine-learning/python-efficient-text-data-cleaning/
    # text normalisation (lower-case, etc)
    text = text.lower()
    # ignore non-ascii characters
    text = text.encode('ascii', 'ignore').decode()
    # keep only words
    text = re.sub(r'[^a-z\s]', '', text)
    # trim spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


df['clean_text'] = df['Article'].dropna().apply(clean_text)

# tokenisation & stop-word removal
df['tokens'] = df['clean_text'].apply(lambda x: [t for t in word_tokenize(x) if t not in STOPWORDS])


# vocabulary limiting/filtering: for a more concise dictionary apply lemmatisation
# https://www.kaggle.com/code/kishalmandal/all-about-stemming-and-lemmatization-cleaning lemmatisation favoured over
# stemming because we want to learn word relationships. stemming removes such relationships and removes meaning (e.g.
# program vs programmer -> both same stem(program), but have different meanings).
# https://www.geeksforgeeks.org/python/python-lemmatization-with-nltk/

# lemmatisation using spaCy
def lemmatise_tokens(tokens):
    doc = nlp(" ".join(tokens))
    return [token.lemma_ for token in doc]


df['lemmas'] = df['tokens'].progress_apply(lemmatise_tokens)

# vocabulary creation - most common 10k words and mapped to indices
all_lemmas = [lemma for doc in df['lemmas'] for lemma in doc]
lemma_freq = Counter(all_lemmas)

most_common = lemma_freq.most_common(MAX_VOCAB_SIZE - 1)  # -1 for UNK
vocab = {word: idx + 1 for idx, (word, _) in enumerate(most_common)}  # start from 1
vocab[UNK_TOKEN] = 0  # UNK mapped to 0


# indexing tokens into numerical indices based on the constructed vocabulary
def convert_to_indices(lemmas):
    return [vocab.get(token, vocab[UNK_TOKEN]) for token in lemmas]


df['lemmas_indices'] = df['lemmas'].apply(convert_to_indices)

# print some info
print(f"Total business articles processed: {len(df)}")

# total number of tokens
total_tokens = sum(len(lemmas) for lemmas in df['lemmas'])
print(f"Total tokens: {total_tokens}")

# vocabulary size
print(f"Vocabulary size: {len(vocab)}")

# top 20 most common words
print("\nTop 20 most frequent lemmas:")
for word, freq in Counter([lemma for doc in df['lemmas'] for lemma in doc]).most_common(20):
    print(f"{word:<15} {freq}")

# keep only df["lemmas"] and df["lemmas_indices"]
df = df[['lemmas', 'lemmas_indices']]

# save
df.to_csv(f"Data/preprocessed_{news_type}_articles.csv", index=False)
df.to_pickle(f"Data/preprocessed_{news_type}_articles.pkl")

with open(f"Data/vocab_{news_type}_dict.pkl", "wb") as f:
    pickle.dump(vocab, f)
