import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# load the df
print('Loading data...')
df = pd.read_csv('diga_translated.csv')

# do the same with the english version
df_en = df [['app_name', 'developer_info', 'indication_codes', 'indication_names_translated', 'app_description_translated']]

# create a dictionary with row index as key and all the other columns as values as a single value
df_dict_en = df_en.T.to_dict('list')

df_dict_en = {k: ' '.join(v) for k, v in df_dict_en.items()}

print('Loading model...')
# load the German model
model2 = SentenceTransformer("sentence-transformers/msmarco-distilbert-dot-v5") #english
model1 = SentenceTransformer('all-MiniLM-L6-v2')


print('Encoding data...')
entries_eng = model1.encode(list(df_dict_en.values()))

print('Ready to search!')
def english_top3(input_text, dictionary, entries = entries_eng):
    input_embedding = model1.encode([input_text])
    similarity2 = np.dot(input_embedding, entries.T)
    indices = np.argsort(similarity2, )[0][-3:][::-1]
    return df_en.iloc[indices]

if __name__ == "__main__":

    # repeat input until user enters 'exit'
    input_text = input("Please describe your concerns: ")

    while input_text != 'quit':
        results = english_top3(input_text, df_dict_en)
        for i in range(3):
            print('')
            print('Result ', i+1, ':')
            print('--------------------')
            print('App name: ', results['app_name'].iloc[i])
            print('Developer: ', results['developer_info'].iloc[i])
            print('Indication code: ', results['indication_codes'].iloc[i])
            print('Indikation description: ', results['indication_names_translated'].iloc[i])
            print('App description: ', results['app_description_translated'].iloc[i])
            print('')
        print('----------------------------------------')
        print('----------------------------------------')
        input_text = input("Please describe your concerns: ")
    