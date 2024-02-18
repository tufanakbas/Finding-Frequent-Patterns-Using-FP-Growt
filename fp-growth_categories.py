import os #for file operations
import csv #for processing on csv
import nltk #natural language processing
import string #for string operations
import pandas as pd #csv read and write operations
from nltk.corpus import stopwords #for language processing
from nltk.tokenize import word_tokenize #for language processing
from mlxtend.frequent_patterns import fpgrowth #for fp-growth algorithm
from mlxtend.preprocessing import TransactionEncoder #to change the data format

# NLTK
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words("turkish"))


def create_output_folder():
    # create a output folder
    output_folder = 'output_folder'
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def preprocess_text(text):
    # remove punc and nums
    text = text.translate(str.maketrans("", "", string.punctuation + string.digits))

    text = text.lower()

    # remove turkish stopwords
    words = word_tokenize(text, language="turkish")
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]

    # seperate words by blank
    preprocessed_text = " ".join(filtered_words)

    return preprocessed_text


def txt_to_csv(root_folder, output_folder):
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)

        if os.path.isdir(folder_path):
            # create csv file
            csv_file_path = os.path.join(output_folder, f"{folder_name}.csv")

            with open(csv_file_path, "w", newline="", encoding="utf-8") as csv_file:
                csv_writer = csv.writer(csv_file)

                # label
                csv_writer.writerow(["Dosya Adı", "İçerik"])

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
    words_to_remove = ["bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz", "on", "a'nî", "ama", "amma",
                       "ancak", "belki", "bile", "çünkü", "da", "de", "dahi", "demek", "dışında", "eğer", "encami",
                       "fakat", "gâh", "gelgelelim", "gibi", "hâlbuki", "hatta", "hem", "ile", "ille velakin",
                       "ille velâkin", "imdi", "kâh", "kaldı ki", "karşın", "ki", "lakin", "madem", "mademki",
                       "maydamı", "meğerki", "meğerse", "ne var ki", "neyse", "oysa", "oysaki", "ve", "velakin",
                       "velev", "velhâsıl", "velhâsılıkelâm", "veya", "veyahut", "ya da", "yahut", "yalıňız", "yalnız",
                       "yani", "yok", "yoksa", "zira", "göre", "kadar"]  # Silmek istediğiniz kelimelerin listesi
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


def process_csv_file(input_file, output_folder):
    # remove_single_letter_words
    output_file_1 = remove_single_letter_words(input_file, os.path.join(output_folder, 'output_1.csv'), 'İçerik')

    # remove_two_letter_words
    output_file_2 = remove_two_letter_words(output_file_1, os.path.join(output_folder, 'output_2.csv'), 'İçerik')

    # remove_specific_words
    output_file_3 = remove_specific_words(output_file_2, os.path.join(output_folder, 'output_3.csv'), 'İçerik')

    # separate_words_with_comma
    output_file_4 = separate_words_with_comma(output_file_3, os.path.join(output_folder, 'output_4.csv'), 'İçerik')

    return output_file_4


def apply_fp_growth(input_csv, output_csv):
    df = pd.read_csv(input_csv)

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

    # write to csv
    sorted_frequent_itemsets.to_csv(output_csv, index=False)


if __name__ == "__main__":

    root_folder = "./news"  # main folder path

    # creating output folder
    output_folder = create_output_folder()

    txt_to_csv(root_folder, output_folder)

    input_folder = output_folder
    output_folder1 = input_folder

    # for all files
    file_list = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

    for file_name in file_list:
        input_file = os.path.join(input_folder, file_name)
        output_file = process_csv_file(input_file, output_folder1)

        # rename output files
        output_csv = os.path.join(output_folder1, f"fp-growth_output_{file_name}")

        # write to csv
        apply_fp_growth(output_file, output_csv)

