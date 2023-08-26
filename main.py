import csv
import pandas as pd
import os
import numpy as np
import json
from pre_process import ExtractFeature, Tokenizer
from oclip.src.clip.model import oCLIP
import argparse
import torch
import torchvision.transforms as T

def extract_features(path, save_dir, batch_size, pre_fn, model, device):

    extractor = ExtractFeature(path, save_dir, batch_size, pre_fn, model, device)
    extractor.extract_features()

    texts = extractor.query_texts()

    return texts

def perform_query(queries, gts, model, tokenizer, feature_path, device):

    photo_features = np.load(os.path.join(feature_path, 'features.npy'))
    ids = list(pd.read_csv(os.path.join(feature_path, 'ids.csv'))['photo_id'])
    
    top1_count = 0
    top3_count = 0
    top5_count = 0
    top10_count = 0
    for idx, query in enumerate(queries):
        
        with torch.no_grad():
            query_encoded = model.encode_text(tokenizer.char_token([query]).to(device), None)
            query_encoded /= query_encoded.norm(dim=-1, keepdim=True)
            query_encoded = query_encoded.cpu().numpy()

        similarities = list((query_encoded @ photo_features[:,0,:].T).squeeze())
        best_photos = sorted(zip(similarities, range(photo_features.shape[0])), \
            key=lambda x: x[0], reverse=True)

        top1_match = best_photos[0][1]
        top3_match = [item[1] for item in best_photos[0:3]]
        top5_match = [item[1] for item in best_photos[0:5]]
        top10_match = [item[1] for item in best_photos[0:10]]
        predict_top1 = ids[top1_match]
        predict_top3 = [ids[i] for i in top3_match]
        predict_top5 = [ids[i] for i in top5_match]
        predict_top10 = [ids[i] for i in top10_match]
        gt_item = [item.split('/')[-1] for item in gts[idx].split(" ")]

        if predict_top1 in gt_item:
            top1_count += 1
        if set(predict_top3).intersection(gt_item):
            top3_count += 1
        if set(predict_top5).intersection(gt_item):
            top5_count += 1
        if set(predict_top10).intersection(gt_item):
            top10_count += 1            

    acc_t1 = top1_count / len(queries)
    acc_t3 = top3_count / len(queries)
    acc_t5 = top5_count / len(queries)
    acc_t10 = top10_count / len(queries)

    return acc_t1, acc_t3, acc_t5, acc_t10

def val_transform(im, image_resolution=512):
    normalize = T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    transform = T.Compose([
        T.ToTensor(),
        normalize,
    ])

    w, h = im.size
    if max(w, h) > image_resolution: 
        ratio = float(image_resolution) / max(w, h)
        w, h = int(w * ratio), int(h * ratio)
    images = transform(im.resize((w, h)))

    images_ = torch.zeros((3, image_resolution, image_resolution))
    mask_ = torch.ones((image_resolution, image_resolution), dtype=torch.bool)
    images_[: images.shape[0], : images.shape[1], : images.shape[2]].copy_(images)
    mask_[: images.shape[1], :images.shape[2]] = False
    mask_ = mask_[::32, ::32]

    return images_.unsqueeze(0), mask_.unsqueeze(0)

def val_transform_simple(im):

    normalize = T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    transform = T.Compose([
        T.ToTensor(),
        normalize,
    ])

    images = transform(im)

    return images.unsqueeze(0)

def load_oclip(device):

    model_path = "weights/RN50_synthtext.pt"
    model_config_file = "oclip/src/training/model_configs/RN50.json"

    with open(model_config_file, 'r') as f:
        model_info = json.load(f)    
    state_dict = torch.load(model_path, map_location="cpu")
    state_dict = state_dict['state_dict']
    state_dict_ = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model = oCLIP(False, **model_info).to(device)
    model.eval()
    model.load_state_dict(state_dict_)

    return model 

def extract_gts(path):

    df = pd.read_csv(path, sep='\t', quoting=csv.QUOTE_NONE)

    queries = df['query'].tolist()
    gts = df['filename'].tolist()

    # gts = [item.split("/")[-1] for item in gts]

    return queries, gts

def test_totalText():

    parser = argparse.ArgumentParser(description="performe text query with oclip")
    parser.add_argument("--dataset_name", type=str, default='totalText')
    parser.add_argument("--data_path", type=str, default='data/weak/totalText/totalText_test_texts.csv')
    parser.add_argument("--gt_path", type=str, default='data/weak/totalText/totalText_test_query.csv')
    parser.add_argument("--save_dir", type=str, default='data/features/totalText/image_features')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--char_dict_path", type=str, default='data/origin/char_dict.pkl')
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_oclip(device)
    char_tokenizer = Tokenizer(args.char_dict_path)
    extract_features(args.data_path, args.save_dir, \
        args.batch_size, val_transform, model, device)
    
    queries, gts = extract_gts(args.gt_path)
    acc_top1, acc_top3, acc_top5, acc_top10 = perform_query(queries, gts, model, \
        char_tokenizer, args.save_dir, device)

    print(f"acc in dataset {args.dataset_name} is top1:{acc_top1:.3f}, \
        top3:{acc_top3:.3f}, top5:{acc_top5:.3f},  top10:{acc_top10:.3f}")

def test_ic15():

    parser = argparse.ArgumentParser(description="performe text query with oclip")
    parser.add_argument("--dataset_name", type=str, default='ic15')
    parser.add_argument("--data_path", type=str, default='data/weak/ic15/IC15_test_texts.csv')
    parser.add_argument("--gt_path", type=str, default='data/weak/ic15/ic15_test_query.csv')
    parser.add_argument("--save_dir", type=str, default='data/features/ic15/image_features')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--char_dict_path", type=str, default='data/origin/char_dict.pkl')
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_oclip(device)
    char_tokenizer = Tokenizer(args.char_dict_path)
    extract_features(args.data_path, args.save_dir, \
        args.batch_size, val_transform, model, device)
    
    queries, gts = extract_gts(args.gt_path)
    acc_top1, acc_top3, acc_top5, acc_top10 = perform_query(queries, gts, model, \
        char_tokenizer, args.save_dir, device)

    print(f"acc in dataset {args.dataset_name} is top1:{acc_top1:.3f}, \
        top3:{acc_top3:.3f}, top5:{acc_top5:.3f},  top10:{acc_top10:.3f}")

def test_svt():

    parser = argparse.ArgumentParser(description="performe text query with oclip")
    parser.add_argument("--dataset_name", type=str, default='svt')
    parser.add_argument("--data_path", type=str, default='data/weak/svt1/svt_test_texts.csv')
    parser.add_argument("--gt_path", type=str, default='data/weak/svt1/svt_test_query.csv')
    parser.add_argument("--save_dir", type=str, default='data/features/svt/image_features')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--char_dict_path", type=str, default='data/origin/char_dict.pkl')
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_oclip(device)
    char_tokenizer = Tokenizer(args.char_dict_path)
    extract_features(args.data_path, args.save_dir, \
        args.batch_size, val_transform, model, device)
    
    queries, gts = extract_gts(args.gt_path)
    acc_top1, acc_top3, acc_top5, acc_top10 = perform_query(queries, gts, model, \
        char_tokenizer, args.save_dir, device)

    print(f"acc in dataset {args.dataset_name} is top1:{acc_top1:.3f}, \
        top3:{acc_top3:.3f}, top5:{acc_top5:.3f},  top10:{acc_top10:.3f}")

def test_iiit():

    parser = argparse.ArgumentParser(description="performe text query with oclip")
    parser.add_argument("--dataset_name", type=str, default='totalText')
    parser.add_argument("--data_path", type=str, default='data/weak/IIIT_STR_V1.0/iiit_test_texts.csv')
    parser.add_argument("--gt_path", type=str, default='data/weak/IIIT_STR_V1.0/iiit_test_query.csv')
    parser.add_argument("--save_dir", type=str, default='data/features/IIIT_STR_V1/image_features')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--char_dict_path", type=str, default='data/origin/char_dict.pkl')
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_oclip(device)
    char_tokenizer = Tokenizer(args.char_dict_path)
    extract_features(args.data_path, args.save_dir, \
        args.batch_size, val_transform, model, device)
    
    queries, gts = extract_gts(args.gt_path)
    acc_top1, acc_top3, acc_top5, acc_top10 = perform_query(queries, gts, model, \
        char_tokenizer, args.save_dir, device)

    print(f"acc in dataset {args.dataset_name} is top1:{acc_top1:.3f}, \
        top3:{acc_top3:.3f}, top5:{acc_top5:.3f},  top10:{acc_top10:.3f}")

if __name__ == '__main__':

    # test_totalText()
    # test_ic15()
    # test_svt()
    test_iiit()