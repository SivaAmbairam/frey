import csv
import queue
import re
import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import ElementClickInterceptedException
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import random
from module_package import *
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from selenium.common.exceptions import ElementClickInterceptedException, TimeoutException
import logging
import glob
import concurrent.futures
from queue import Queue


# Setup logging
log_dir = r'Output/temp'
log_file = 'web_scraping_frey.log'

os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, log_file)

logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s %(message)s')


nltk.data.path.append(r'C:\Users\svc_webscrape\AppData\Roaming\nltk_data')

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
stop_words = set(stopwords.words('english'))

def process_flinn_product(original_flinn_row, flinn_word_set, frey_products, threshold, driver, flinn_csv, frey_csv):
    original_flinn_product = original_flinn_row['Flinn_product_name']
    flinn_product_id = original_flinn_row['Flinn_product_id']
    flinn_row = flinn_csv[flinn_csv['Flinn_product_id'] == flinn_product_id]
    try:
        desc_name = flinn_row.iloc[0]['Flinn_product_desc']
    except:
        desc_name = ''
    key_name = original_flinn_product
    best_match = None
    best_match_score = 0

    for original_frey_row, frey_word_set in frey_products:
        combined_similarity = word_similarity(flinn_word_set, frey_word_set)
        if 0.3 <= threshold <= 0.4:
            combined_similarity = float(re.search(r'\\d*\.\\d*', str(combined_similarity)).group())
            if combined_similarity == threshold:
                product_ids = fetch_frey_product_ids(driver, key_name)
                if not product_ids:
                    if combined_similarity >= best_match_score:
                        best_match_score = combined_similarity
                        best_match = original_frey_row
                    continue
                for product_id in product_ids:
                    frey_row = frey_csv[frey_csv['Frey_product_id'] == product_id]
                    if not frey_row.empty:
                        frey_title = frey_row.iloc[0]['Frey_product_name']
                        frey_description = frey_row.iloc[0]['Frey_product_desc']
                        title_similarity_score = calculate_similarity(key_name, frey_title, pooling_strategy='mean')
                        description_similarity_score = calculate_similarity(desc_name, frey_description, pooling_strategy='mean')
                        combined_similarity_score = (title_similarity_score + description_similarity_score) / 2

                        if combined_similarity_score >= best_match_score:
                            best_match_score = combined_similarity_score
                            best_match = original_frey_row
                break
        else:
            if combined_similarity >= best_match_score:
                best_match_score = combined_similarity
                best_match = original_frey_row

    return original_flinn_row, best_match, best_match_score


def remove_stop_words(sentence):
    if isinstance(sentence, float):
        return ''
    words = word_tokenize(sentence.lower())
    filtered_words = [word for word in words if word not in stop_words]
    filtered_sentence = ' '.join(filtered_words)
    return filtered_sentence


def get_sentence_embedding(sentence, pooling_strategy='mean'):
    filtered_sentence = remove_stop_words(sentence)
    inputs = tokenizer(filtered_sentence, return_tensors='pt', truncation=True, padding=True, max_length=128)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    last_hidden_state = outputs.last_hidden_state

    if pooling_strategy == 'mean':
        sentence_embedding = torch.mean(last_hidden_state, dim=1)
    elif pooling_strategy == 'cls':
        sentence_embedding = last_hidden_state[:, 0, :]
    elif pooling_strategy == 'max':
        sentence_embedding = torch.max(last_hidden_state, dim=1)[0]
    else:
        raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")

    sentence_embedding = torch.nn.functional.normalize(sentence_embedding, p=2, dim=1)
    return sentence_embedding


def calculate_similarity(sentence1, sentence2, pooling_strategy='max'):
    embedding1 = get_sentence_embedding(sentence1, pooling_strategy)
    embedding2 = get_sentence_embedding(sentence2, pooling_strategy)
    similarity = cosine_similarity(embedding1.numpy(), embedding2.numpy())
    return similarity[0][0]


color_names = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'brown', 'black', 'white', 'gray', 'silver']


def clean_text(text):
    for color in color_names:
        text = re.sub(rf'\b{color}\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d+(\.\d+)?\s*(mL|mm)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d+\b', '', text)
    return text.strip()


def get_word_set(text):
    return set(word for word in re.split(r'\W+', text) if word)


def word_similarity(set1, set2):
    return len(set1 & set2) / len(set1 | set2)



def fetch_frey_product_ids(driver, key_name):
    try:
        driver.get('https://www.schoolspecialty.com/')
        search_element = WebDriverWait(driver, 30).until(
            EC.element_to_be_clickable((By.NAME, 'searchTerm'))
        )
        search_element.send_keys(key_name)

        search_button = WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.XPATH, "//a[@class='submitButton']"))
        )

        try:
            driver.execute_script("arguments[0].scrollIntoView(true);", search_button)
            WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, "//a[@class='submitButton']")))
            search_button.click()
        except ElementClickInterceptedException:
            driver.execute_script("arguments[0].click();", search_button)

        time.sleep(random.randint(1, 30))
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        results_number = soup.find_all('div',  class_='product_SKU')
        product_ids = [single_number.text.split(': ', 1)[-1].strip() for single_number in results_number]
        return product_ids
    except TimeoutException as e:
        logging.error(f"TimeoutException: {e}")
        driver.save_screenshot('timeout_exception_screenshot.png')
    except Exception as e:
        logging.error(f"An error occurred: {e}")


def match_products(flinn_products, frey_products, initial_threshold, threshold_decrement, output_folder, batch_size=100, num_threads=4):
    start_time = time.time()
    total_comparisons = 0

    matched_products = []
    threshold = initial_threshold
    output_folder_path = os.path.join(rf'D:\svc_webscrape\Deployment 2/Scrapping Scripts/Output/temp/{output_folder}')
    os.makedirs(output_folder_path, exist_ok=True)

    flinn_file_path = os.path.join(r'D:\svc_webscrape\Deployment 2/Scrapping Scripts/Output/Flinn_Products.csv')
    frey_file_path = os.path.join(r'D:\svc_webscrape\Deployment 2/Scrapping Scripts/Output/Frey_Products.csv')

    flinn_csv = pd.read_csv(flinn_file_path)
    frey_csv = pd.read_csv(frey_file_path)
    options = Options()
    options.add_argument("--headless")

    result_queue = Queue()

    def worker(batch):
        local_results = []
        driver = webdriver.Chrome(options=options)
        for original_flinn_row, flinn_word_set in batch:
            result = process_flinn_product(original_flinn_row, flinn_word_set, frey_products, threshold, driver, flinn_csv, frey_csv)
            local_results.append(result)
        driver.quit()
        result_queue.put(local_results)

    while threshold >= 0:
        print(f"Matching products with threshold: {threshold:.2f}")
        output_file = os.path.join(output_folder_path, f"FlinnVsFrey_{threshold:.2f}.csv")

        with open(output_file, 'w', newline='', encoding='utf-8') as master_file:
            writer = csv.writer(master_file)
            writer.writerow(['Flinn_product_category', 'Flinn_product_sub_category', 'Flinn_product_id', 'Flinn_product_name',
                             'Flinn_product_quantity', 'Flinn_product_price', 'Flinn_product_url', 'Flinn_image_url',
                             'Frey_product_category', 'Frey_product_sub_category', 'Frey_product_id', 'Frey_product_name',
                             'Frey_product_quantity', 'Frey_product_price', 'Frey_product_url', 'Frey_image_url',
                             'Frey_Match_Score'])

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                for i in range(0, len(flinn_products), batch_size):
                    batch = flinn_products[i:i+batch_size]
                    executor.submit(worker, batch)

                executor.shutdown(wait=True)

            while not result_queue.empty():
                results = result_queue.get()
                for result in results:
                    original_flinn_row, best_match, best_match_score = result

                    flinn_colors = [color for color in color_names if re.search(rf'\b{color}\b', original_flinn_row['Flinn_product_name'], re.IGNORECASE)]
                    frey_colors = [color for color in color_names if best_match and re.search(rf'\b{color}\b', best_match['Frey_product_name'], re.IGNORECASE)]

                    flinn_ml_mm = re.findall(r'\b\d+(\.\d+)?\s*(mL|mm)\b', original_flinn_row['Flinn_product_name'], re.IGNORECASE)
                    frey_ml_mm = re.findall(r'\b\d+(\.\d+)?\s*(mL|mm)\b', best_match['Frey_product_name'], re.IGNORECASE) if best_match else []

                    if best_match_score >= threshold:
                        if set(flinn_colors) == set(frey_colors) and set(flinn_ml_mm) == set(frey_ml_mm):
                            writer.writerow([
                                original_flinn_row['Flinn_product_category'], original_flinn_row['Flinn_product_sub_category'],
                                original_flinn_row['Flinn_product_id'], original_flinn_row['Flinn_product_name'],
                                original_flinn_row['Flinn_product_quantity'], original_flinn_row['Flinn_product_price'],
                                original_flinn_row['Flinn_product_url'], original_flinn_row['Flinn_image_url'],
                                best_match['Frey_product_category'], best_match['Frey_product_sub_category'],
                                best_match['Frey_product_id'], best_match['Frey_product_name'],
                                best_match['Frey_product_quantity'], best_match['Frey_product_price'],
                                best_match['Frey_product_url'], best_match['Frey_image_url'], best_match_score
                            ])
                            matched_products.append((original_flinn_row, best_match, best_match_score))
                        elif set(flinn_colors) == set(frey_colors):
                            writer.writerow([
                                original_flinn_row['Flinn_product_category'], original_flinn_row['Flinn_product_sub_category'],
                                original_flinn_row['Flinn_product_id'], original_flinn_row['Flinn_product_name'],
                                original_flinn_row['Flinn_product_quantity'], original_flinn_row['Flinn_product_price'],
                                original_flinn_row['Flinn_product_url'], original_flinn_row['Flinn_image_url'],
                                best_match['Frey_product_category'], best_match['Frey_product_sub_category'],
                                best_match['Frey_product_id'], best_match['Frey_product_name'],
                                best_match['Frey_product_quantity'], best_match['Frey_product_price'],
                                best_match['Frey_product_url'], best_match['Frey_image_url'], best_match_score
                            ])
                            matched_products.append((original_flinn_row, best_match, best_match_score))
                        else:
                            writer.writerow([
                                original_flinn_row['Flinn_product_category'], original_flinn_row['Flinn_product_sub_category'],
                                original_flinn_row['Flinn_product_id'], original_flinn_row['Flinn_product_name'],
                                original_flinn_row['Flinn_product_quantity'], original_flinn_row['Flinn_product_price'],
                                original_flinn_row['Flinn_product_url'], original_flinn_row['Flinn_image_url'],
                                best_match['Frey_product_category'], best_match['Frey_product_sub_category'],
                                best_match['Frey_product_id'], best_match['Frey_product_name'],
                                best_match['Frey_product_quantity'], best_match['Frey_product_price'],
                                best_match['Frey_product_url'], best_match['Frey_image_url'], best_match_score
                            ])
                            matched_products.append((original_flinn_row, best_match, best_match_score))
                    else:
                        writer.writerow([
                            original_flinn_row['Flinn_product_category'], original_flinn_row['Flinn_product_sub_category'],
                            original_flinn_row['Flinn_product_id'], original_flinn_row['Flinn_product_name'],
                            original_flinn_row['Flinn_product_quantity'], original_flinn_row['Flinn_product_price'],
                            original_flinn_row['Flinn_product_url'], original_flinn_row['Flinn_image_url'],
                            '', '', '', 'No good match found (Low match score)', '', '', '', '', 0
                        ])

                    total_comparisons += 1
                    if total_comparisons % 100 == 0:
                        elapsed_time = time.time() - start_time
                        print(f"Processed {total_comparisons} comparisons in {elapsed_time:.2f} seconds")
                        print(f"Current threshold: {threshold:.2f}")
                        print(f"Matches found: {len(matched_products)}")
                        print("---")

        threshold = round(threshold - threshold_decrement, 2)

    total_time = time.time() - start_time
    print(f"\nMatching completed:")
    print(f"Total comparisons: {total_comparisons}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Comparisons per second: {total_comparisons / total_time:.2f}")
    print(f"Total matches found: {len(matched_products)}")

    return matched_products


flinn_file_path = os.path.join(r'D:\svc_webscrape\Deployment 2/Scrapping Scripts/Output/Flinn_Products.csv')
frey_file_path = os.path.join(r'D:\svc_webscrape\Deployment 2/Scrapping Scripts/Output/Frey_Products.csv')


with open(flinn_file_path, 'r', encoding='utf-8') as flinn_file, open(frey_file_path, 'r', encoding='utf-8') as frey_file:
    flinn_reader = csv.DictReader(flinn_file)
    frey_reader = csv.DictReader(frey_file)

    flinn_products = [(row, get_word_set(clean_text(row['Flinn_product_name']))) for row in flinn_reader]
    frey_products = [(row, get_word_set(clean_text(row['Frey_product_name']))) for row in frey_reader]


initial_threshold = 0.8
threshold_decrement = 0.01
output_folder = 'FlinnVsFrey'


output_files = match_products(flinn_products, frey_products, initial_threshold, threshold_decrement, output_folder, batch_size=10000, num_threads=20)


output_csv_dir = r'D:\svc_webscrape\Deployment 2/Scrapping Scripts/Output/temp/FlinnVsFrey/*.csv'
csv_files = glob.glob(output_csv_dir)
csv_files = [file for file in csv_files if not file.endswith('Matched_Products.csv')]
dfs = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    dfs.append(df)
merged_df = pd.concat(dfs, ignore_index=True)
merged_df.drop_duplicates(subset=['Flinn_product_name', 'Flinn_product_id'], keep='first', inplace=True)
merged_output_file = r'D:\svc_webscrape\Deployment 2/Scrapping Scripts/Output/temp/FlinnVsFrey/Matched_Products.csv'
os.makedirs(os.path.dirname(merged_output_file), exist_ok=True)
merged_df.to_csv(merged_output_file, index=False)
print(f"Merged {len(csv_files)} CSV files into '{merged_output_file}'.")