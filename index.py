#%%
import json
import re
from collections import defaultdict
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from Models.word2vec import word2vecSG


with open("./movies.json", "r") as f:
    movies_json = json.load(f)

def tokenize(text):
    if isinstance(text,str):
        text = text.lower()  
        return re.findall(r'\b\w+\b', text)
    else: return []


def invert_index(docs: list) -> dict:
    """Builds an inverted index from the parsed JSON file"""
    inverted_index = defaultdict(list)
    for record in tqdm(docs, total=len(docs), desc="Building Inverted Index"):

        #Tokenize movie and plot for efficient search
        name_tokens = tokenize(record['movie_name'])

        summary_tokens = tokenize(record['summary'])

        doc_tokens = name_tokens + summary_tokens
            
        for token in doc_tokens:
            inverted_index[token].append(record)
        
    return inverted_index

def basic_search(query: str, inverted_index: dict) -> dict:
    """Returns basic search results based on the inverted index"""

    #Tokenize the query
    query_tokens = tokenize(query)
    search_results = {}

    for token in tqdm(query_tokens, total=len(query_tokens), desc="Going through query tokens"):

        if token in inverted_index:
            #run through all the movie fields
            for movie in inverted_index[token]:
                search_results[movie['movie_name']] = {
                                            'release_date' : movie['release_date'],
                                            'box_office' : movie['box_office'],
                                            'plot_summary' : movie['summary']
                                                    }
    
    print(f"    {len(search_results)} results found")
                
    return search_results


def tfidf_search(docs: list[str], query: str):
    """Converts a list of docs into TFID vocab, transforms the query and computes CS"""

    #create a summary to move_id index
    summary_movie = {}
    for record in docs:
        if ('summary' in record) and isinstance(record['summary'],str):
            summary_movie[record['summary']] = record

    summaries = [s for s in list(summary_movie.keys()) if isinstance(s,str)]

    #Create the sparse matrix
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(summaries)

    #Transform the query
    query_tfidf = vectorizer.transform([query])

    #Compute cosine similarities 
    cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix)

    cosine_similarities = cosine_similarities.flatten()

    #Get top 5 matches
    top5_matches = np.argsort(cosine_similarities)[-5:][::-1] 

    #Get the equivalent summaries
    print("The top 5 matches are")
    for i, index in enumerate(top5_matches):
        print (f"{i+1} : {summary_movie[summaries[index]]}\n\n")


def get_word_embeddings(docs: list[str]) -> object:
    """Get word embeddings of the entire corpus of movie summaries"""
    #Extract all summaries into a list of sentences
    extracted_sentences = []
    for movie in tqdm(docs, total = len(docs), desc="Extracting words from movie summaries"):
        if 'summary' in movie and isinstance(movie['summary'],str):
            extracted_sentences.extend([m for m in movie['summary'].split('.') if m != ''])
    if len(extracted_sentences) > 0:
        #Reduced the sentence to first 100 for a test run
        word2vec = word2vecSG(extracted_sentences[:101])
    else:
        print("Error - extracted sentences are empty")

    #Get the woord2vec embeddings
    word2vec.fit(epochs = 200)
    return word2vec




#%%

if __name__ == "__main__":

    #Basic inverted search
    inverted_index = invert_index(movies_json)
    search_query = "Interstellar and Kubrick"
    basic_search(search_query, inverted_index)

    #TFIDF search
    tfidf_search(movies_json, search_query)



# %%
