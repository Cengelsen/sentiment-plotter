import re
import os
import glob
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json
import yake
from datetime import datetime
from textblob import TextBlob 
from sklearn.metrics.pairwise import cosine_similarity


# Load Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Universal Sentence Encoder (USE)
def calculate_sts_similarity(text1, text2):
    
    # Get embeddings for the texts
    embedding1 = embed([text1])[0].numpy()
    embedding2 = embed([text2])[0].numpy()

    # Calculate cosine similarity
    similarity = cosine_similarity([embedding1], [embedding2])

    return similarity

# Creates the initial data structure
proposals = []

utterance_list = []

# Start of timer
start_time = time.time()

# READ FILE SECTION
directory = "."
markdown_files = glob.glob(os.path.join(directory, "**/*.md"), recursive=True)

def isDumbFile(file):
    
    if "toc.md" in file.lower() \
    or "summary.md" in file.lower():
        return True
    else:
        return False

markdown_files = [file for file in markdown_files if isDumbFile(file) == False]

def process_markdown_file(markdown_file):
    markdown_file = markdown_file.lstrip("./")
    with open(markdown_file, "r", encoding="utf-8") as file:
        return file.read()

def extract_timestamp(markdown_file):
    month_dict = {
        "jan": 1,
        "feb": 2,
        "february": 2,
        "mar": 3,
        "apr": 4,
        "april": 4,
        "may": 5,
        "jun": 6,
        "july": 7,
        "aug": 8,
        "sept": 9,
        "september": 9,
        "oct": 10,
        "october": 10,
        "nov": 11,
        "dec": 12,
        "december": 12
    }

    # searches through markdown files only
    title_match = re.search(r"(\w+)-(\d+)\.md", markdown_file)
    document_date = markdown_file

    # Builds the timestamp based on the file-name
    if title_match:
        document_month = title_match.group(1).lower()
        document_day = title_match.group(2)

        if document_month in month_dict:
            month_number = month_dict[document_month]
            current_year = datetime.now().year
            document_date = f"{current_year}-{month_number:02d}-{document_day}"
    else:
        document_date = "N/A"

    return document_date

def isDumbTitle(title):
    if "organizational" in title.lower() \
        or "meeting" in title.lower() \
        or "ending....." in title.lower() \
        or "agenda" in title.lower() \
        or "others" in title.lower() \
        or "consensus" in title.lower() \
        or "summary" in title.lower() \
        or "conclusion" in title.lower() \
        or "closing" in title.lower() \
        or "approval" in title.lower() \
        or "needs work" in title.lower() \
        or "welcome" in title.lower() \
        or "announcement" in title.lower() \
        or "resolution" in title.lower() \
        or "introduction" in title.lower() \
        or "minutes" in title.lower() \
        or "secretary" in title.lower() \
        or "secretariat" in title.lower() \
        or "housekeeping" in title.lower() \
        or "intro" in title.lower() \
        or "election" in title.lower() \
        or "proposals" in title.lower() \
        or "committee" in title.lower() \
        or "reminder" in title.lower() \
        or title.lower() == "":

        return True
    else:
        return False


def areSimilar(text):

    prop_titles = [proposal["title"] for proposal in proposals]
    full_texts = [proposal["full text"] for proposal in proposals]

    zipped = zip(prop_titles, full_texts)

    # index, title of proposal, degree of similarity
    most_similar = (0, "", 0)

    for zip_index, (title, prop_text) in enumerate(zipped, start=0):

        similarity = calculate_sts_similarity(text, prop_text)

        if similarity >= most_similar[2]:
            most_similar = (zip_index, title, similarity)

    return most_similar


# Use map for parallel processing and list comprehension to process all files concurrently
markdown_texts = [process_markdown_file(file) for file in markdown_files]
timestamps = [extract_timestamp(file) for file in markdown_files]

title_w_proptext = []
dumb_titles = []
smart_titles = []

# Iterate over processed files
for markdown_text, current_date in zip(markdown_texts, timestamps):
    # Defines the pattern of the proposal sections
    proposal_section_pattern = re.compile(r"##\s(.*)", re.MULTILINE)
    proposal_titles = re.findall(proposal_section_pattern, markdown_text)

    proposal_text = ""

    utterance_list = []

    current_title = ""

    proposal_parts = []

    # For-løkke som henter ut all tekst mellom hver proposal og spytter dem inn i en ny liste
    for prop_index, section_title in enumerate(proposal_titles, start=1):

        current_title = section_title.lower()

        if isDumbTitle(current_title) == True:
            dumb_titles.append(current_title)
            continue

        else:
            smart_titles.append(current_title)

            current_title = section_title.strip()

            section_start = markdown_text.find(section_title)

            if section_start != -1:
                proposal_text = markdown_text[section_start + len(section_title):].strip()

            # Defines the pattern of features to ignore: the presenter and the slides.
            presenter_pattern = r"^Presenter: .+$"
            proposal_text = re.sub(presenter_pattern, "", proposal_text, flags=re.MULTILINE).strip()

            slides_pattern = r"- \[(.*?)\]\((.*?)\)"
            proposal_text = re.sub(slides_pattern, "", proposal_text) 

            title_w_proptext.append((current_title, proposal_text))

            # Defines the pattern that the utterances adhere to
            utterance_pattern = r"([A-Z]{2,3}):([^\n]*|$)"

            # Extracts all the utterances
            extracted_utterances = re.findall(utterance_pattern, proposal_text)

            # Create a list to store utterance objects
            utterances = []
            for utt_index, ext_utterance in enumerate(extracted_utterances, start=1):

                speakers = []

                speaker, utterance_text = ext_utterance
                speakers.append(speaker)
                split_utterance = re.split(r'[.!?]', utterance_text)  # Split into sentences

                # Create a list to store sentence objects
                sentences = []

                # entire utterance joined together
                concat_utterance = "".join(utterance_text) 

                kw_extractor = yake.KeywordExtractor(lan="en", 
                                     n=3, 
                                     dedupLim=0.9, 
                                     dedupFunc='seqm', 
                                     windowsSize=1, 
                                     top=10)

                utt_keywords = kw_extractor.extract_keywords(concat_utterance)

                for sent_index, sentence_text in enumerate(split_utterance, start=1):

                    # Creates a sentence object
                    sentence = {
                        "sentence_number": sent_index,
                        "text": sentence_text,
                        "polarity": TextBlob(sentence_text).polarity,
                        "subjectivity": TextBlob(sentence_text).subjectivity
                    }
                    sentences.append(sentence)
                
                # Creates an utterance object
                utterance = {
                    "utterance_number": utt_index,
                    "timestamp": current_date,
                    "sentences": sentences,  # Store list of sentences directly
                    "polarity": TextBlob(concat_utterance).polarity,
                    "subjectivity": TextBlob(concat_utterance).subjectivity,
                    "keywords": utt_keywords

                }
                utterances.append(utterance)

                utterance_list.append(concat_utterance)

            clean_prop_text = "".join(utterance_list)

            # Creates a proposal object
            proposal = {
                #"proposal number": prop_index,
                "title": current_title,
                "timestamp": current_date,
                "utterances": utterances,  # list of all utterance objects
                "full text": clean_prop_text
            }

            most_similar = areSimilar(proposal["full text"])

            print(f"current title is:\n {proposal['title']}\n")

            if most_similar[2] >= 0.98:            

                if len(proposals) == 0:
                    proposals.append(proposal)
                else:

                    print(f"most similar index: {most_similar[0]}")
                    print(f"most similar title: {most_similar[1]}")
                    print(f"degree of similarity: {most_similar[2]}\n")

                    proposals[most_similar[0]]["utterances"] + proposal["utterances"]

            else:
                print("nothing similar enough\n")
                print(f"highest similarity: {most_similar[2]}")
                print(f"most similar prop: {most_similar[1]}")
                proposals.append(proposal)
                print(f"nr. of proposals: {len(proposals)}\n") 


print("Text extraction completed")

# CREATE PLOTS FOR SENTIMENTS

save_folder = "sentiment_plots"
os.makedirs(save_folder, exist_ok=True)

for idx, proposal in enumerate(proposals, start=1):
    data = [utt["polarity"] for utt in proposal["utterances"]]            
    
    # Extracts the utterances with peaking sentiment polarity
    peaking_utterances = []
    for utt in proposal["utterances"]:
        if utt["polarity"] > 0.75:
            peaking_utterances.append(utt)

        elif utt["polarity"] < -0.75:
            peaking_utterances.append(utt)


    x_values = range(1, len(proposal["utterances"]) + 1)
    
    # Plotting the values
    plt.plot(x_values, data, label="Line Plot")

    # Calculate the upper limit for x-axis ticks based on the number of utterances
    max_utterances = len(proposal["utterances"])
    upper_limit = max_utterances + (10 - (max_utterances % 10))  # Ensure upper limit is a multiple of 10

    # Specify the granularity of the x-axis ticks
    plt.xticks(np.arange(1, upper_limit, 10))

    # Set static y-axis bounds
    plt.ylim(-1.0, 1.0)

    # Calculate the width of the plot based on the number of markers
    marker_count = len(np.arange(1, upper_limit, 10))
    plot_width = max(marker_count * 0.5, 6.0)  # Adjust the multiplier as needed

    # Set the width of the plot
    plt.gcf().set_size_inches(plot_width, 5)  # Adjust the height (second parameter) as needed


    # Adding labels and title
    plt.xlabel('Utterances')
    plt.ylabel('Sentiment')
    plt.title(proposal["title"])

    # Create a directory for each proposal title
    proposal_folder = os.path.join(save_folder, proposal["title"].replace("/", "_"))
    os.makedirs(proposal_folder, exist_ok=True)

    # Save the plot to a file (PNG format) inside the proposal folder
    filename = os.path.join(proposal_folder, f"{proposal['title'].replace('/', '_')}_{idx}.png")
    plt.savefig(filename)

    # Write peaking utterances to a JSON file inside the proposal folder
    json_file_path = os.path.join(proposal_folder, "peaking_utterances.json")
    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(peaking_utterances, json_file, indent=2)

    # Write the utterances of the current proposal to a JSON file inside the proposal folder
    json_file_path = os.path.join(proposal_folder, "utterances.json")
    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(proposal["utterances"], json_file, indent=2)

    # Clear the current figure for the next iteration
    plt.clf()

# Write proposals to a JSON file
json_file_path = os.path.join(save_folder, "proposals.json")
with open(json_file_path, "w", encoding="utf-8") as json_file:
    json.dump(proposals, json_file, indent=2)


# Calculates the total run time
end_time = time.time()
execution_time = end_time - start_time

# Convert the execution time to hours, minutes, and seconds
hours, remainder = divmod(execution_time, 3600)
minutes, seconds = divmod(remainder, 60)

# Prints the execution time in a human-readable format
print(f"Execution Time: {int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds")
