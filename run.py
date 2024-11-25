import json
import nltk
from nltk.corpus import stopwords
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
import re
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

nltk.download('stopwords')
stop_words = set(stopwords.words('german'))

def preprocess_and_prepare_for_lda(texts):
    # Preprocessing and tokenization
    processed_texts = []
    for text in texts:
        # lowercase
        text = text.lower()

        # remove urls and special characters
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # tokenize and remove stopwords
        tokens = [word for word in text.split() if word not in stop_words]
        processed_texts.append(tokens)

    # create dictionary and bow corpus for lda and filter extreme values from dictionary
    dictionary = Dictionary(processed_texts)
    dictionary.filter_extremes(no_below=20, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    return dictionary, corpus, processed_texts

if __name__ == "__main__":

    # load tripadvisor reviews
    with open("data/dataset_tripadvisor-reviews_2024-11-10_10-40-29-916.json", "r", encoding='utf-8') as f:
        tripadvisor_reviews = json.load(f)

    # load google reviews
    with open("data/dataset_Google-Maps-Reviews-Scraper_2024-11-10_10-55-33-122.json", "r", encoding='utf-8') as f:
        google_reviews = json.load(f)

    # Check, if reviews were loaded correctly:
    print(f"{len(tripadvisor_reviews)} Tripadvisor reviews were successfully loaded.")
    print(f"{len(google_reviews)} Google reviews were successfully loaded.")

    '''
    368 Tripadvisor reviews were successfully loaded.
    274 Google reviews were successfully loaded.
    '''

    # extract texts
    tripadvisor_reviews_texts = [review['text'] for review in tripadvisor_reviews if review['text']]
    google_reviews_texts = [review['text'] for review in google_reviews if review['text']]
    all_reviews = tripadvisor_reviews_texts + google_reviews_texts
    new_list = []

    # replace backspaces
    for review in all_reviews:
        new_list.append(review.replace("\n", ""))

    all_reviews = new_list

    # save review texts for GPT-based review analysis
    with open("text_corpus.txt", "w", encoding='utf-8') as f:
        f.write(str(all_reviews))

    #extract german reviews for topic modeling
    tripadvisor_reviews_german_texts = [review['text'] for review in tripadvisor_reviews if review['text'] and review['lang'] == "de"]

    # Preprocess and prepare data for topic modeling
    dictionary, corpus, processed_texts = preprocess_and_prepare_for_lda(tripadvisor_reviews_german_texts)

    # train LDA model
    lda_model = LdaModel(corpus=corpus,
                         id2word=dictionary,
                         num_topics=15,
                         random_state=42)

    # save topic model as html
    lda_vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(lda_vis_data, 'lda_visualization.html')

