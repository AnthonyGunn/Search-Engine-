#Jules-Andrei Labador 20516006
#Anthony Gunn 93114604
#Rayden Wang 17732162

import os
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer #AKA Porter2
from nltk.corpus import stopwords
import json
import pickle
from time import perf_counter
from collections import defaultdict
import math


STEMMER = SnowballStemmer("english") #Porter2 stemmer
STOP_WORDS = set(stopwords.words("english"))

def search(query: str, index: 'file', term_index: dict, term_set: set, url_mapping: dict, result_count: int) -> None:
    
    '''
    When searching for a query such as 'machine learning' search for all the docs that include 'machine' and 'learning'.
    Then, find all the docs where those 2 words overlap.
    '''
    start_time = perf_counter()

    global STOP_WORDS
    global STEMMER

    # Split query up into keywords
    query_terms = set(word_tokenize(query))
    print(f"🟢 Searching for query_terms: {query_terms}")

    # [actual entry, current position when searching]
    index_entries_with_pos = []

    score = defaultdict(int)

    stop_word_count = 0
    found_stop_words = set()

    for term in query_terms:
        if term in STOP_WORDS:
            stop_word_count += 1
            found_stop_words.add(term)

    if (len(query_terms) > 7 or stop_word_count > 4) and stop_word_count > 0:
        amount_to_discard = stop_word_count - 4
        stop_word_dfs = set()
        for stop_word in found_stop_words:
            index.seek(term_index[stop_word])
            stop_word_dfs.add((int(index.readline().split()[1]), stop_word))

        for i in range(0, amount_to_discard):
            highest_doc_count_stop_word = ""
            highest_doc_count = 0
            for stop_word_with_df in stop_word_dfs:
                index.seek(term_index[stop_word])
                if stop_word_with_df[0] > highest_doc_count:
                    highest_doc_count = stop_word_with_df[0]
                    highest_doc_count_stop_word = stop_word_with_df

            stop_word_dfs.remove(highest_doc_count_stop_word)
            query_terms.remove(highest_doc_count_stop_word[1])

    for term in query_terms:
        for split_term in term.split("-"):
            # Stem the term so it matches up with the ones in 'index'
            stemmed_term = STEMMER.stem(split_term)

            if stemmed_term not in term_set:
                break

            # Copy the docs from 'index' to 'index_entries_with_pos' using the stemmed term 'stemmed_term'
            index.seek(term_index[stemmed_term])
            index_entries_with_pos.append([index.readline().split(), 2]) # [actual entry, current position when searching]

    else:
        # Get the term with the smallest doc_count
        lowest_doc_count_entry_index = index_entries_with_pos.index(min(index_entries_with_pos, key = lambda entry : int(entry[0][1])))
        
        # # Iterate through all doc_ids in the index entry with the lowest doc_count
        print(f"Gathering indexes...")

        seen_urls = set()

        # current index position in the entry with lowest doc count
        current_pos1 = index_entries_with_pos[lowest_doc_count_entry_index][1] 
        while current_pos1 < len(index_entries_with_pos[lowest_doc_count_entry_index][0]):
            current_doc_id = int(index_entries_with_pos[lowest_doc_count_entry_index][0][current_pos1])

            if(url_mapping[current_doc_id] in seen_urls):
                current_pos1 += 8
                continue
            seen_urls.add(url_mapping[current_doc_id])
            
            # Iterate through the other entries
            for i in range(0, len(index_entries_with_pos)):

                # For each entry, iterate through its postings to try to find current_doc_id
                current_pos2 = index_entries_with_pos[i][1] # current index position in the current entry

                # Loop through each entry (each entry is an array)
                while current_pos2 < len(index_entries_with_pos[i][0]):

                    if int(index_entries_with_pos[i][0][current_pos2]) >= current_doc_id:                  
                        break
                #------------------------------------------------------
                    current_pos2 += 8 # offset of 8
                    index_entries_with_pos[i][1] = current_pos2

                else: # reached the end of the entry and current_doc_id was not found
                    break

                if int(index_entries_with_pos[i][0][current_pos2]) != current_doc_id: # went past current_doc_id without finding it
                    break

            else:
                # Document contains all terms
                for entry in index_entries_with_pos:
                    doc_index = int(entry[1])
                    number_of_docs = int(entry[0][1])

                    N_val = len(url_mapping) 

                    # document frequency value
                    df_val = number_of_docs

                    # term frequency value
                    tf_val = int(entry[0][doc_index + 1])
                    if tf_val > 50:
                        tf_val = tf_val * 0.50

                    title_score = int(entry[0][doc_index + 2]) * 50
                    h1_score = int(entry[0][doc_index + 3]) * 10
                    h2_score = int(entry[0][doc_index + 4]) * 3
                    h3_score = int(entry[0][doc_index + 5]) * 2
                    bold_score = int(entry[0][doc_index + 6]) * 1.050
                    tags_sum = title_score + h1_score + h2_score + h3_score + bold_score

                    total_score = (1 + math.log(tf_val + tags_sum, 10)) * math.log(N_val / df_val, 10)
                    score[current_doc_id] += total_score

            current_pos1 += 8

    print(f"\tCollecting top {result_count} URLs for '{query}'")
    top_documents = sorted(score, key=score.get, reverse=True)[:result_count]
    results = dict()

    end_time = perf_counter()
    print(f"\tSearch took {(end_time - start_time)*1000}ms")
    #print(score)

    for count, i in enumerate(top_documents):
        print(f"\t {count}: {url_mapping[i]}")
        results[i] = url_mapping[i]

    # Create the 'results' directory
    # if not os.path.exists("results/"):
    #     os.mkdir("results/")

    # Write to the 'results' directory with search results
    # with open(f'results/result-{query}.json', 'w') as f:
    #     json.dump(results, f)



if __name__ == '__main__':
    with open("index.txt", "r", encoding="utf-8") as index, open("term_index.pkl", "rb") as term_file, open("url_mapping.pkl", "rb") as url_file:
        term_index = pickle.load(term_file)
        url_mapping = pickle.load(url_file)

        term_set = set(term_index.keys())

        # with open("term_index.json", 'w') as f:
        #     json.dump(term_index, f)
        # with open("url_mapping.json", 'w') as f:
        #     json.dump(url_mapping, f)
        
        while True:
            query_term = input("Enter search query: ")
            if query_term != "":
                search(query_term, index, term_index, term_set, url_mapping, 10)