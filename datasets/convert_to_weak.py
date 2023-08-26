import csv
import pandas as pd
import pickle
import random
from tqdm import tqdm
import os
import scipy.io, scipy.ndimage
import argparse
from xml.etree.ElementTree import ElementTree


class Converter:

    def __init__(self, name, path, output_dir):

        self.name = name
        self.path = path
        self.output_dir = output_dir
        self.train = False

    def convert(self):

        if not os.path.isdir(os.path.join(self.output_dir)):
            os.makedirs(os.path.join(self.output_dir))

        if self.name == 'SynthText':
            self.convert_synthText()
        elif self.name == 'IC15':
            self.convert_ic15()
        elif self.name == 'totalText':
            self.convert_totalText()
        elif self.name == 'iiit':
            self.convert_iiit()
        elif self.name == 'svt':
            self.convert_svt()

        if not os.path.isfile(os.path.join(os.path.dirname(self.path), \
            'char_dict.pkl')):

            with open(os.path.join(os.path.dirname(self.path),\
                'char_dict.pkl'), 'wb') as f:
                char_list = list(range(33, 127))
                pickle.dump(char_list, f)
        
        print(f"convert {self.name} dataset successfully")
            
    def convert_synthText(self):
                
        synth_data = scipy.io.loadmat(os.path.join(self.path, 'gt.mat'))
        imnames = synth_data['imnames'][0]
        text = synth_data['txt'][0]

        #self.to_table(imnames, text, self.output_dir)
        i_range = range(len(text))
        print(f'num_images:{len(i_range)}')

        data_out = ['filepath\ttitle\n']
        for i, im_idx in tqdm(enumerate(i_range)):
            i_img_name = os.path.join(self.path, 'Images', str(imnames[im_idx][0]))
            i_text = text[im_idx]

            word_list = '\n'.join(i_text).split()
            random.shuffle(word_list)

            data_out += [f"{i_img_name}\t{' '.join(word_list)}\n"]

        self.train = True
        with open(os.path.join(self.output_dir, f"{self.name}_{'train' if self.train else 'test'}_texts.csv"), 'w') as f:
            f.writelines(data_out)

    def convert_ic15(self):
        
        if self.train == True:
            image_foler = os.path.join(self.path, 'train_images')
            gts_folder = os.path.join(self.path, 'train_gts')
        else:
            image_foler = os.path.join(self.path, 'test_images')
            gts_folder = os.path.join(self.path, 'test_gts')

        img_names = sorted([os.path.join(image_foler, name) for name in os.listdir(image_foler)])
        gt_path_list = sorted([os.path.join(gts_folder, name) for name in os.listdir(gts_folder)])

        print(f'num_images:{len(img_names)}')

        data_out = ['filepath\ttitle\n']
        for i_img_name, gt in tqdm(zip(img_names, gt_path_list)):
            
            with open(gt, 'rb') as f:
                raw_lines = f.readlines()
            
            i_text = [line.decode('utf-8').strip().split(',')[-1] for line in raw_lines]
            word_list = [word for word in i_text if word!='###']
            if word_list == []:
                continue
            random.shuffle(word_list)
            data_out += [f"{i_img_name}\t{' '.join(word_list)}\n"]

        with open(os.path.join(self.output_dir, f"{self.name}_{'train' if self.train else 'test'}_texts.csv"), 'w') as f:
            f.writelines(data_out)

    def convert_totalText(self):
        if self.train == True:
            image_foler = os.path.join(self.path, 'Images', 'Train')
            gts_folder = os.path.join(self.path, 'total_text_labels', 'train_gts')
        else:
            image_foler = os.path.join(self.path, 'Images', 'Test')
            gts_folder = os.path.join(self.path, 'total_text_labels', 'test_gts')

        img_names = sorted([os.path.join(image_foler, name) for name in os.listdir(image_foler)])
        gt_path_list = sorted([os.path.join(gts_folder, name) for name in os.listdir(gts_folder)])

        print(f'num_images:{len(img_names)}')

        data_out = ['filepath\ttitle\n']
        for i_img_name, gt in tqdm(zip(img_names, gt_path_list)):
            
            with open(gt, 'rb') as f:
                raw_lines = f.readlines()
            
            i_text = [line.decode('utf-8').strip().split(',')[-1] for line in raw_lines]
            word_list = [word for word in i_text if word!='###']
            if word_list == []:
                continue
            random.shuffle(word_list)
            data_out += [f"{i_img_name}\t{' '.join(word_list)}\n"]

        with open(os.path.join(self.output_dir, f"{self.name}_{'train' if self.train else 'test'}_texts.csv"), 'w') as f:
            f.writelines(data_out)
        
    def convert_iiit(self):
        
        image_folder = os.path.join(self.path, 'imgDatabase')
        gt_path = os.path.join(self.path, 'data.mat')
        img_names = [os.path.join(self.path, 'imgDatabase', name) for name in os.listdir(image_folder)]
        data = scipy.io.loadmat(gt_path)
        query_list = [str(data['data'][0,i][0][0][0][0]) for i in range(data['data'].shape[1])]
        target_list = [data['data'][0,i][1] for i in range(data['data'].shape[1])]

        print(f'num_images:{len(img_names)}')

        data_out = ['query\tfilepath\n']
        for idx, query in tqdm(enumerate(query_list)):
            
            i_img_name_raw = target_list[idx]
            i_img_name = [os.path.join(self.path,item[0][0]) for item in i_img_name_raw]
            
            data_out += [f"{query}\t{' '.join(i_img_name)}\n"]

        with open(os.path.join(self.output_dir, f"{self.name}_{'train' if self.train else 'test'}_query.csv"), 'w') as f:
            f.writelines(data_out)

    def parse_xml_file(self, gt_path):
        datas = []
        tree = ElementTree()
        tree.parse(gt_path)
        image_list = []
        texts_list = []
        for object_ in tree.findall("image"):
            image_name = object_.find("imageName").text
            image_list.append(image_name)
            texts = []
            for text_object in object_.findall("taggedRectangles/taggedRectangle"):
                text = text_object.find("tag").text
                texts.append(text)

            texts_list.append(texts)

        return image_list, texts_list


    def convert_svt(self):

        self.image_folder = os.path.join(self.path, 'img')
        if self.train:
            self.gt_path = os.path.join(self.path, 'train.xml')
        else:
            self.gt_path = os.path.join(self.path, 'test.xml')

        img_names, texts = self.parse_xml_file(self.gt_path)

        print(f'num_images:{len(img_names)}')

        data_out = ['filepath\ttitle\n']
        for i_img_name, gt in tqdm(zip(img_names, texts)):
            
            word_list = [word for word in gt if word!='###']
            if word_list == []:
                continue
            random.shuffle(word_list)
            data_out += [f"{os.path.join(self.path,i_img_name)}\t{' '.join(word_list)}\n"]

        with open(os.path.join(self.output_dir, f"{self.name}_{'train' if self.train else 'test'}_texts.csv"), 'w') as f:
            f.writelines(data_out)
        



def test_synthText():
    
    parser = argparse.ArgumentParser('prepare the standard synthText dataset into texts: image pairs')
    parser.add_argument('--name', type=str, default='synthText', help='name of the dataset')
    parser.add_argument('--path', type=str, default='./data/origin/SynthText', help='path to the synthText dataset')
    parser.add_argument('--output', type=str, default='./data/weak/SynthText', help='path to the output dataset')
    args = parser.parse_args()

    synth_convert = Converter(args.name, args.path, args.output)
    synth_convert.convert()


def main():

    # synthText_convert = Converter('SynthText', './data/origin/SynthText', './data/weak/SynthText')
    # synthText_convert.convert()

    ic15_convert = Converter('IC15', './data/origin/ic15', './data/weak/ic15')
    ic15_convert.convert()

    #totalText_convert = Converter('totalText', './data/origin/totalText', './data/weak/totalText')
    #totalText_convert.convert()

    # iiit_convert = Converter('iiit', './data/origin/IIIT_STR_V1.0', './data/weak/IIIT_STR_V1.0')
    # iiit_convert.convert()

    # svt_convert = Converter('svt', './data/origin/svt1', './data/weak/svt1')
    # svt_convert.convert()

def iiit_utils():
    iiit_path = "data/weak/IIIT_STR_V1.0/iiit_test_query.csv"
    df = pd.read_csv(iiit_path, delimiter='\t', quoting=csv.QUOTE_NONE)
    texts = df['query'].tolist()
    images = df['filepath'].tolist()

    images_list = [item.split(" ") for item in images]

    tmp_img_list = []
    for item in images_list:
        tmp_img_list.extend(item)

    new_images_list = list(set(tmp_img_list))
    data_out = ['filepath\ttitle\n']
    for img in new_images_list:
        words_list = []

        for idx, item_ in enumerate(images_list):
            
            if img in item_:
                words_list.append(texts[idx])
        data_out += [f"{img}\t{' '.join(words_list)}\n"]

    with open("data/weak/IIIT_STR_V1/iiit_test_texts.csv", 'w') as f:
        f.writelines(data_out)


if __name__ == '__main__':

    # main()
    iiit_utils()