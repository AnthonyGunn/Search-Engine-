#Jules-Andrei Labador 20516006
#Anthony Gunn 93114604
#Rayden Wang 17732162

import os
import re
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer #AKA Porter2
from collections import defaultdict, namedtuple
import pickle
import json
from urllib.parse import urldefrag
import multiprocessing
from glob import glob
from time import perf_counter


'''
INDEX STRUCTURE (subject to change)
During construction:
Index is a dict<term: str, posting_list: list>
posting_list contains 2-tuples (DocPosting) (doc_id: int, amounts_list: list)
amounts_list contains 6 items [amount: int, title_amount: int, h1_amount: int, h2_amount: int, h3_amount: int, bold_amount: int]
Format on disk for each term (one per line):
term doc_count doc_id_1 amount title_amount h1_amount h2_amount h3_amount bold_amount , doc_id_2 amount title_amount h1_amount h2_amount h3_amount bold_amount , 
Explanation:
    term: the string of the term
    doc_count: total # docs in this posting list (aka len(posting_list))
    
    doc_id_#: the document id
    amount: TOTAL amount of that term in the document (including the ones that were in tags)
    (tag)_amount: amount of that term in a specific tag
    ,: denotes the end of a amounts_list
    each one of these is seperated by a space
IMPORTANT CHANGE: next_doc_offset was removed, since its always going to be 8
'''


seen_urls = set()

TEXT_TAG_BLACKLIST = {"[document]","noscript","meta","head","input","script","style"}
STEMMER = SnowballStemmer("english") #Porter2 stemmer



DocPostings = namedtuple("DocPosting", "doc_id amounts_list")
CombinedIndexData = namedtuple("CombinedIndexData", "index_fp term_index url_mapping")



# Parses all directories in the 'DEV' folder
def generate_index(path: str = "DEV/", process_count: int = os.cpu_count()) -> None:
    print(f"starting indexing of files in {path} using {process_count} process(es)")
    start_time = perf_counter()

    filepaths = glob(path + "**/*.json", recursive=True) # Gets all json files in the given directory and its subdirectories
    amount_per_process = len(filepaths)//process_count # Number of files each process (excluding the last) will handle 

    jobs_args = [] # list which holds tuples of args for each job
    
    for i in range(0, process_count-1):
        split_filepaths = filepaths[(i*amount_per_process):((i+1)*amount_per_process)]
        jobs_args.append((split_filepaths, i, i*amount_per_process))
    # last process does all remaining files
    split_filepaths = filepaths[((process_count-1)*amount_per_process):]
    jobs_args.append((split_filepaths, process_count-1, (process_count-1)*amount_per_process))

    with multiprocessing.Pool(processes=process_count, maxtasksperchild=1) as pool:
        all_combined_index_data = pool.starmap(generate_partial_indexes, jobs_args) # start processes

        # merge all info from each process after they complete
        final_term_index = merge_partial_indexes(all_combined_index_data, "index.txt")
        with open("term_index.pkl", "wb") as out_file:
            pickle.dump(final_term_index, out_file, pickle.HIGHEST_PROTOCOL)

        merge_partial_url_mappings(all_combined_index_data)

    end_time = perf_counter()
    print(f"indexing finished in approx {end_time - start_time}s")



def generate_partial_indexes(filepaths: list, process_num: int, starting_doc_id: int) -> CombinedIndexData:
    url_mapping = dict()
    current_doc_id = starting_doc_id

    amount_per_dump = len(filepaths)//3

    term_indexes = []

    for i in range(0, 2):
        index = defaultdict(list)
        for filepath in filepaths[(i*amount_per_dump):((i+1)*amount_per_dump)]:
            parse_file(filepath, index, url_mapping, current_doc_id)
            current_doc_id += 1
        term_indexes.append(dump_partial_index(index, process_num, i))
        print(f"process #{process_num} progress: {i+1}/3 dumped")
    # last dump contains all remaining files
    index = defaultdict(list)
    for filepath in filepaths[(2*amount_per_dump):]:
        parse_file(filepath, index, url_mapping, current_doc_id)
        current_doc_id += 1
    term_indexes.append(dump_partial_index(index, process_num, 2))

    print(f"process #{process_num} progress: 3/3 dumped. done.")

    # pair the index filepath with the term index
    index_term_index_pairs = [CombinedIndexData(index_fp=f"partial_indexes/index{process_num}-{i}.txt", term_index=term_indexes[i], url_mapping=None) for i in range(0,3)]
    
    # merge the 3 indexes and return
    final_partial_term_index = merge_partial_indexes(index_term_index_pairs, f"partial_indexes/index{process_num}.txt")
    return CombinedIndexData(index_fp=f"partial_indexes/index{process_num}.txt", term_index=final_partial_term_index, url_mapping=url_mapping)



def parse_file(filepath: str, index: dict, url_mapping: dict, current_doc_id: int) -> None:
    global TEXT_TAG_BLACKLIST
    global STEMMER

    with open(filepath, "r", encoding="utf-8") as f:
        if ".DS_Store" in filepath:
            return

        ### parse json ###
        parsed_json = json.loads(f.read())
        url = parsed_json["url"]

        ### Defrag URL. If it's already in 'seen_urls', exit out. We don't need to parse it. ###
        defragged_url = urldefrag(url)[0]
        if defragged_url in seen_urls:
            return
        
        seen_urls.add(defragged_url)

        ### remove html comments ###
        filtered_html = re.sub(r"<!--(\n|.)*?-->", "", parsed_json["content"].lower())

        ### collect and process textual content ###
        soup = BeautifulSoup(filtered_html, "lxml")
        text_list = soup.find_all(string=lambda s : s.parent.name not in TEXT_TAG_BLACKLIST)

        # Default value for any dict entry is a DocPostings with the specified default values
        word_dict = defaultdict(lambda : DocPostings(doc_id=current_doc_id, amounts_list=[0,0,0,0,0,0]))

        for text in text_list:
            ### set tag flags (1 if tag is present, 0 if not) ###
            text_tags = set([parent.name.lower() for parent in text.parents])
            title_tag = 0
            h1_tag = 0
            h2_tag = 0
            h3_tag = 0
            bold_tag = 0
            if "title" in text_tags:
                title_tag = 1
            if "h1" in text_tags:
                h1_tag = 1
            if "h2" in text_tags:
                h2_tag = 1
            if "h3" in text_tags or "h4" in text_tags or "h5" in text_tags or "h6" in text_tags:
                h3_tag = 1
            if "strong" in text_tags or "b" in text_tags:
                bold_tag = 1
            
            ### add data to word_dict ###
            clean_text = re.sub(r"</?\w+>", "", text)
            words = word_tokenize(clean_text)
            for word in words:
                split_words = word.split("-")
                for split_word in split_words:
                    if split_word.replace("'", "").isalnum(): # allows things like contractions, which has apostrophes
                        #For any term, add increment proper amounts
                        stemmed_word = STEMMER.stem(split_word)
                        word_dict[stemmed_word].amounts_list[0] += 1
                        word_dict[stemmed_word].amounts_list[1] += title_tag
                        word_dict[stemmed_word].amounts_list[2] += h1_tag
                        word_dict[stemmed_word].amounts_list[3] += h2_tag
                        word_dict[stemmed_word].amounts_list[4] += h3_tag
                        word_dict[stemmed_word].amounts_list[5] += bold_tag

        ### add word_dict data to index ###
        update_index(word_dict, defragged_url, index, url_mapping, current_doc_id)



def update_index(word_dict: dict, url: str, index: dict, url_mapping: dict, doc_id: int) -> None:
    #TODO: implement n-grams maybe
    for word, doc_postings in word_dict.items():
        # Append DocPostings to the list of given term in the index
        index[word].append(doc_postings)

    url_mapping[doc_id] = url



#returns a dict that maps terms to their position in the partial index txt file
def dump_partial_index(index: dict, process_num: int, dump_number: int) -> dict:
    if not os.path.exists("partial_indexes/"):
        os.mkdir("partial_indexes/")

    term_index = dict()

    # term doc_count doc_id_1 amount title_amount h1_amount h2_amount h3_amount bold_amount , doc_id_2 amount title_amount h1_amount h2_amount h3_amount bold_amount ,
    with open(f"partial_indexes/index{process_num}-{dump_number}.txt", "w", encoding="utf-8") as out_file:
        for term, posting_list in index.items():
            line = f"{term} {len(posting_list)} " # "term doc_count "

            for doc_postings in posting_list:
                line += f"{doc_postings.doc_id} " # "doc_id_# "

                for amount in doc_postings.amounts_list:
                    line += f"{amount} " # "amount " or "(tag)_amount"

                line += ", "

            term_index[term] = out_file.tell() #index the index - map terms to their posting list positions in the index file
            out_file.write(f"{line}\n")

    return term_index



#returns a dict that maps terms to their position in the merged index txt file
def merge_partial_indexes(combined_index_data_list: list, output_filepath: str) -> dict:
    term_index = dict()
    with open(output_filepath, "w", encoding="utf-8") as out_file:
        opened_indexes = []
        term_set = set()

        #make set containing terms across all indexes
        for i in range(0, len(combined_index_data_list)):
            opened_indexes.append(open(combined_index_data_list[i].index_fp, "r", encoding="utf-8"))
            for term in combined_index_data_list[i].term_index.keys():
                term_set.add(term)

        for term in term_set:
            split_lines = []

            #if the term is in an index, read the line for it and save it as an array of tokens
            for i in range(0, len(combined_index_data_list)):
                if term in combined_index_data_list[i].term_index:
                    opened_indexes[i].seek(combined_index_data_list[i].term_index[term])
                    split_lines.append(opened_indexes[i].readline().split())
                else:
                    split_lines.append([])

            #sum the doc_counts
            summed_doc_count = 0
            for i in range(0,len(combined_index_data_list)):
                if len(split_lines[i]) != 0:
                    summed_doc_count += int(split_lines[i][1])

            #merge the postings lists across all indexes for that term
            merged_line = f"{term} {summed_doc_count} "
            for i in range(0,len(combined_index_data_list)):
                if len(split_lines[i]) != 0:
                    merged_line += " ".join(split_lines[i][2:len(split_lines[i])])
                    merged_line += " "

            term_index[term] = out_file.tell() #index the index - map terms to their posting list positions in the index file
            out_file.write(f"{merged_line}\n")

    return term_index



def merge_partial_url_mappings(combined_index_data_list: list) -> None:
    for i in range(1, len(combined_index_data_list)):
        combined_index_data_list[0].url_mapping.update(combined_index_data_list[i].url_mapping)

    with open("url_mapping.pkl", "wb") as out_file:
        pickle.dump(combined_index_data_list[0].url_mapping, out_file, pickle.HIGHEST_PROTOCOL)





if __name__ == '__main__':
    # Required for nltk, only needs to be run once.
    # nltk.download('popular')

    # set folder path to whatever. this function will use as many processes as cpu cores by default.
    generate_index(path="DEV-analyst/")
