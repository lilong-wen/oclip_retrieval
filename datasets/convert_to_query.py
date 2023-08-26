import os
from tqdm import tqdm
import csv
import string
#import nltk
#from nltk.corpus import stopwords
import pandas as pd


class ToQueryForm:

    def __init__(self,name, path):

        self.name = name
        self.path = path
        self.img_key = 'filepath'
        self.text_key = 'title'
        self.train = False
        
        with open('data/stopwords_en.txt', 'r') as f:
            self.stop_words_en = f.readlines()

        df = pd.read_csv(path, sep='\t', quoting=csv.QUOTE_NONE)
        self.images = df[self.img_key].tolist()
        self.texts = df[self.text_key].tolist()

    def convert_to_query(self):

        all_query = []
        all_texts = []

        for i_texts in self.texts:

            i_texts = [i.translate(str.maketrans('', '', string.punctuation)) \
                for i in i_texts.split()]
            i_texts = [text.strip().lower() for text in i_texts if text.lower() \
                not in self.stop_words_en]

            i_texts = [text for text in i_texts if len(text)>2 and text!='']

            if len(i_texts) == 0:
                continue

            all_query.extend(i_texts)
            all_texts.append(i_texts)

        all_query = set(all_query)

        data_out = ['query\tfilename\n']
        for query in tqdm(all_query):

            i_image_name = []
            for i_img, i_texts in zip(self.images, all_texts):
                if query in i_texts:
                    i_image_name.append(i_img)

            data_out+=[f"{query}\t{' '.join(i_image_name)}\n"]

        save_path = os.path.join(os.path.dirname(self.path), \
            f"{self.name}_{'train' if self.train else 'test'}_query.csv")
        with open(save_path, 'w') as f:
            f.writelines(data_out)   


def main():

    # svt_convert = ToQueryForm('svt', 'data/weak/svt1/svt_test_texts.csv')
    # svt_convert.convert_to_query()

    # ic15_convert = ToQueryForm('ic15', 'data/weak/ic15/IC15_test_texts.csv')
    # ic15_convert.convert_to_query()

    # totalText_convert = ToQueryForm('totalText', 'data/weak/totalText/totalText_test_texts.csv')
    # totalText_convert.convert_to_query()

    SynthText_convert = ToQueryForm('SynthText', 'data/weak/SynthText/SynthText_train_texts.csv')
    SynthText_convert.convert_to_query()

if __name__ == '__main__':

    main()