import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# load the df
print('Loading data...')
df = pd.read_csv('diga.csv')

# take each row with app_name, developer_info, indication_codes, indication_names, app_description
df_ger = df[['app_name', 'developer_info', 'indication_codes', 'indication_names', 'app_description']]

# create a dictionary with row index as key and all the other columns as values as a single value
df_dict = df_ger.T.to_dict('list')
df_dict = {k: ' '.join(v) for k, v in df_dict.items()}

print('Loading model...')
# load the German model
model2 = SentenceTransformer("Sahajtomar/German-semantic")

print('Encoding data...')
entries_ger = model2.encode(list(df_dict.values()))

print('Ready to search!')
def german_top(input_text, dictionary, entries = entries_ger):
    input_embedding = model2.encode([input_text])
    similarity2 = np.dot(input_embedding, entries.T)
    # only return similarity scores above 120
    # similarity2 = np.where(similarity2 > 120)
    # print(similarity2)
    # if similarity2[0].max() == 0:
    #     print('No results found')
    #     return
    indices = np.argsort(similarity2)[0][-3:][::-1]
    # print(f'sorted top 5 indices: {indices}')
    # print(f'similarity scores: {similarity2[0][indices]}')
    return df_ger.iloc[indices]

if __name__ == "__main__":

    # repeat input until user enters 'exit'
    input_text = input("Beschreiben Sie bitte ihre Beschwerden: ")

    while input_text != 'quit':
        results = german_top(input_text, df_dict)
        for i in range(3):
            print('')
            print('Result ', i+1, ':')
            print('--------------------')
            print('App Name: ', results['app_name'].iloc[i])
            print('Entwickler: ', results['developer_info'].iloc[i])
            print('Indikation: ', results['indication_codes'].iloc[i])
            print('Indikation: ', results['indication_names'].iloc[i])
            print('Beschreibung: ', results['app_description'].iloc[i])
            print('')
        print('----------------------------------------')
        print('----------------------------------------')
        input_text = input("Beschreiben Sie bitte ihre Beschwerden: ")
    