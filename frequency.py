import pandas as pd
from collections import Counter

def calculate_word_frequencies(input_csv, output_csv):

    df = pd.read_csv(input_csv)

    # select label
    all_text = ','.join(df['İçerik'].astype(str))
    words = all_text.split(',')

    # calculate frequency
    word_frequencies = Counter(words)

    # create DataFrame
    frequency_df = pd.DataFrame(list(word_frequencies.items()), columns=['Word', 'Frequency'])

    # sort
    frequency_df = frequency_df.sort_values(by='Frequency', ascending=False)

    # write to csv
    frequency_df.to_csv(output_csv, index=False)

    print(f"Kelimelerin frekansları '{output_csv}' dosyasına yazıldı.")

# csv names and paths
input_csv_path = './output_4.csv'
output_csv_path = 'word_frequencies1.csv'

# run
calculate_word_frequencies(input_csv_path, output_csv_path)
