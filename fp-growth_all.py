import os #for file operations
import csv #for processing on csv
import nltk #natural language processing
import string #for string operations
import pandas as pd #csv read and write operations
from nltk.corpus import stopwords #for language processing
from nltk.tokenize import word_tokenize #for language processing
from mlxtend.frequent_patterns import fpgrowth #for fp-growth algorithm
from mlxtend.preprocessing import TransactionEncoder #to change the data format


# Installing the Turkish stopwords list using NLTK
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words("turkish"))

def preprocess_text(text):
    # Remove punctuation and numbers
    text = text.translate(str.maketrans("", "", string.punctuation + string.digits))

    text = text.lower()

    # Remove Turkish stopwords
    words = word_tokenize(text, language="turkish")
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]

    # Combine words separated by a space
    preprocessed_text = " ".join(filtered_words)

    return preprocessed_text

def txt_to_csv(root_folder, csv_file):
    with open(csv_file, "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Dosya Adı", "İçerik"])
        for folder_name in os.listdir(root_folder):
            folder_path = os.path.join(root_folder, folder_name)

            # Read the txt files in each subfolder and write them to CSV
            for txt_file_name in os.listdir(folder_path):
                if txt_file_name.endswith(".txt"):
                    txt_file_path = os.path.join(folder_path, txt_file_name)
                    with open(txt_file_path, "r", encoding="utf-8") as txt_file_content:
                        content = txt_file_content.read()
                        preprocessed_content = preprocess_text(content)
                        csv_writer.writerow([txt_file_name, preprocessed_content])

def remove_single_letter_words(input_file, output_file, target_column):
    with open(input_file, 'r', newline='', encoding='utf-8') as csv_input, \
         open(output_file, 'w', newline='', encoding='utf-8') as csv_output:

        csv_reader = csv.DictReader(csv_input)
        fieldnames = csv_reader.fieldnames

        csv_writer = csv.DictWriter(csv_output, fieldnames=fieldnames)
        csv_writer.writeheader()

        for row in csv_reader:
            words = row[target_column].split()
            updated_words = [word for word in words if len(word) > 1]
            row[target_column] = " ".join(updated_words)
            csv_writer.writerow(row)

    return output_file

def remove_two_letter_words(input_file, output_file, target_column):
    with open(input_file, 'r', newline='', encoding='utf-8') as csv_input, \
         open(output_file, 'w', newline='', encoding='utf-8') as csv_output:

        csv_reader = csv.DictReader(csv_input)
        fieldnames = csv_reader.fieldnames

        csv_writer = csv.DictWriter(csv_output, fieldnames=fieldnames)
        csv_writer.writeheader()

        for row in csv_reader:
            words = row[target_column].split()
            updated_words = [word for word in words if len(word) != 2]
            row[target_column] = " ".join(updated_words)
            csv_writer.writerow(row)

    return output_file

def remove_specific_words(input_file, output_file, target_column):
    words_to_remove = ["bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz", "on","a'nî", "ama", "amma",
                       "ancak", "belki", "bile", "çünkü", "da", "de", "dahi", "demek", "dışında", "eğer", "encami", "fakat",
                       "gâh", "gelgelelim", "gibi", "hâlbuki", "hatta", "hem", "ile", "ille velakin", "ille velâkin", "imdi",
                       "kâh", "kaldı ki", "karşın", "ki", "lakin", "madem", "mademki", "maydamı", "meğerki", "meğerse", "ne var ki",
                       "neyse", "oysa", "oysaki", "ve", "velakin", "velev", "velhâsıl", "velhâsılıkelâm", "veya", "veyahut", "ya da",
                       "yahut", "yalıňız", "yalnız", "yani", "yok", "yoksa", "zira","göre", "kadar"]  # List of words want to delete
    with open(input_file, 'r', newline='', encoding='utf-8') as csv_input, \
         open(output_file, 'w', newline='', encoding='utf-8') as csv_output:

        csv_reader = csv.DictReader(csv_input)
        fieldnames = csv_reader.fieldnames

        csv_writer = csv.DictWriter(csv_output, fieldnames=fieldnames)
        csv_writer.writeheader()

        for row in csv_reader:
            words = row[target_column].split()
            updated_words = [word for word in words if word not in words_to_remove]
            row[target_column] = " ".join(updated_words)
            csv_writer.writerow(row)

    return output_file

def separate_words_with_comma(input_file, output_file, target_column):
    with open(input_file, 'r', newline='', encoding='utf-8') as csv_input, \
         open(output_file, 'w', newline='', encoding='utf-8') as csv_output:

        csv_reader = csv.DictReader(csv_input)
        fieldnames = csv_reader.fieldnames

        csv_writer = csv.DictWriter(csv_output, fieldnames=fieldnames)
        csv_writer.writeheader()

        for row in csv_reader:
            words = row[target_column].split()
            row[target_column] = ",".join(words)
            csv_writer.writerow(row)

    return output_file



root_folder = "./news"  # main folder path or name
csv_dosya = "input1.csv"

# txt -> csv
txt_to_csv(root_folder, csv_dosya)

# 4 functions to preprocess the data
input_file = 'input1.csv'

output_file_1 = remove_single_letter_words(input_file, 'output_1.csv', 'İçerik')
output_file_2 = remove_two_letter_words(output_file_1, 'output_2.csv', 'İçerik')
output_file_3 = remove_specific_words(output_file_2, 'output_3.csv', 'İçerik')
output_file_4 = separate_words_with_comma(output_file_3, 'output_4.csv', 'İçerik')

#do not change (output_4 is an output created by other func)
csv_file = 'output_4.csv'
df = pd.read_csv(csv_file)

transactions = df["İçerik"].apply(lambda x: x.split(','))

# Transaction Encoder formatting
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Frequent Pattern Growth
frequent_itemsets = fpgrowth(df_encoded, min_support=0.2, use_colnames=True)
sorted_frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)

sorted_frequent_itemsets['support'] = sorted_frequent_itemsets['support'].apply(lambda x: round(x, 6))

sorted_frequent_itemsets['itemsets'] = sorted_frequent_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))

# output file name
output_csv = "fp-growth_output.csv"

# write to csv
sorted_frequent_itemsets.to_csv(output_csv, index=False)

# print outputs
print(sorted_frequent_itemsets)