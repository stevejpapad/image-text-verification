import os
import clip
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from string import digits


def preprocess_input(path, input_caption, preprocess, device):
    
    text = None
    
    if path:
        image = preprocess(Image.open(path)).unsqueeze(0).to(device)

    else:
        image = None
        
    attempt = 0
    limit = 4

    while True:

        attempt += 1

        if attempt == 1:
            max_len = 70

        elif attempt == 2:
            max_len = 50

        elif attempt == 3:

            print('Try 3')
            max_len = 30

        elif attempt == 4:

            print('Last attempt')
            max_len = 20

        else: 
            break

        try:
            caption = input_caption
            
            if attempt > 3:
                print("Drastic measure", caption)

                remove_digits = str.maketrans('', '', digits)
                caption = caption.translate(remove_digits) 

            caption = ' '.join(caption.split(' ')[:max_len])
            text = clip.tokenize(caption.split('"')[0]).to(device)

            break

        except Exception as ex:
            if limit == attempt:
                break
                
    return image, text


def load_dataset(data_path):
        
        
    print("Load: ", data_path)
    
    if 'EntitySwaps_topic_clip' in data_path:
        ner_train_df = pd.read_csv('VisualNews/train_entity_swap_topic_CLIP.csv', index_col=0)
        ner_valid_df = pd.read_csv('VisualNews/valid_entity_swap_topic_CLIP.csv', index_col=0)
        ner_test_df = pd.read_csv('VisualNews/test_entity_swap_topic_CLIP.csv', index_col=0)

        data = pd.concat([ner_train_df, ner_valid_df, ner_test_df])
        data = data[data.falsified == True]
        
    elif 'EntitySwaps_topic_random' in data_path:
        ner_train_df = pd.read_csv('VisualNews/train_entity_swap_topic.csv', index_col=0)
        ner_valid_df = pd.read_csv('VisualNews/valid_entity_swap_topic.csv', index_col=0)
        ner_test_df = pd.read_csv('VisualNews/test_entity_swap_topic.csv', index_col=0)

        data = pd.concat([ner_train_df, ner_valid_df, ner_test_df])
        data = data[data.falsified == True]
        
    elif 'MISALIGN' in data_path:
        
        train_df = pd.read_csv('Fakeddit/all_samples/all_train.tsv', sep='\t')
        train_df = train_df[train_df['2_way_label'] == 0]
        train_df = train_df[~train_df.clean_title.isna()]
        train_df['id'] = '_TRAIN'
        
        valid_df = pd.read_csv('Fakeddit/all_samples/all_validate.tsv', sep='\t')
        valid_df = valid_df[valid_df['2_way_label'] == 0]
        valid_df = valid_df[~valid_df.clean_title.isna()]
        valid_df['id'] = '_VALID'
        
        test_df = pd.read_csv('Fakeddit/all_samples/all_test_public.tsv', sep='\t')
        test_df = test_df[test_df['2_way_label'] == 0]
        test_df = test_df[~test_df.clean_title.isna()]
        test_df['id'] = '_TEST'
        
        data = pd.concat([train_df, valid_df, test_df])
        data['caption'] = data['clean_title']
                
    elif 'COSMOS' in data_path:
        data = []
        for line in open('COSMOS/cosmos_anns/test_data.json', 'r'):
            data.append(json.loads(line))
    
        data = pd.DataFrame(data)
        data['caption'] = data['caption1']
        data['image_path'] = 'images_test/' + data["img_local_path"]
            
    elif 'VisualNews' in data_path: 
        data = json.load(open(data_path + 'data.json'))
        data = pd.DataFrame(data)
        
    elif 'VERITE' in data_path:
        data = pd.read_csv(data_path + 'VERITE.csv', index_col=0)
        data = pd.DataFrame(data)
        
    elif 'Fakeddit' in data_path:
        train_df = pd.read_csv(data_path + 'all_samples/all_train.tsv', sep='\t')
        valid_df = pd.read_csv(data_path + 'all_samples/all_validate.tsv', sep='\t')
        test_df = pd.read_csv(data_path + 'all_samples/all_test_public.tsv', sep='\t')
        data = pd.concat([train_df, valid_df, test_df])
        data = data[~data.image_url.isna()]
        data = data[~data.clean_title.isna()]
        
        data['image_path'] = 'images/' + data['id'] + ".jpg"
        data["caption"] = data["clean_title"]
        
    return data

def extract_CLIP_features(data_path, output_path, use_image=True, choose_clip_version = "ViT-L/14", choose_gpu = 0):
               
    os.environ["CUDA_VISIBLE_DEVICES"] = str(choose_gpu)
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    model, preprocess = clip.load(choose_clip_version, device=device)    
    
    data = load_dataset(data_path)
    
    save_id = 'id' in data.columns
    model.eval()

    all_ids, all_text_features, all_visual_features = [], [], []

    with torch.no_grad():

        for (row) in tqdm(data.itertuples(), total=data.shape[0]):     
                       
            if not use_image:
                
                caption = row.caption
                image, text = preprocess_input(None, caption, preprocess, device)

                if text != None:
                    text_features = model.encode_text(text)        
                    all_text_features.append(text_features.cpu().detach().numpy()[0])
                    if save_id:
                        all_ids.append(row.id)
            
            else:                
                path =  data_path + row.image_path.split('./')[-1]                

                if os.path.isfile(path):                
                    caption = row.caption
                    image, text = preprocess_input(path, caption, preprocess, device)

                    if text != None:
                        image_features = model.encode_image(image)
                        text_features = model.encode_text(text)        

                        all_text_features.append(text_features.cpu().detach().numpy()[0])
                        all_visual_features.append(image_features.cpu().detach().numpy()[0])

                        if save_id:                                
                            all_ids.append(row.id)

            
    print("Save: ", output_path)
    
    clip_version = choose_clip_version.replace('-', '').replace('/', '')

    all_text_features = np.array(all_text_features).reshape(len(all_text_features),-1)  
    np.save(output_path + "clip_text_embeddings_" + clip_version + ".npy", all_text_features)

    if use_image:
        all_visual_features = np.array(all_visual_features).reshape(len(all_visual_features),-1)
        np.save(output_path + "clip_image_embeddings_" + clip_version + ".npy", all_visual_features) 
    
    if save_id:
        
        if 'MISALIGN' in data_path:
            
            new_ids = []
            for i in range(all_ids.count('_TRAIN')):
                new_ids.append(str(i) + '_TRAIN')

            for i in range(all_ids.count('_VALID')):
                new_ids.append(str(i) + '_VALID')

            for i in range(all_ids.count('_TEST')):
                new_ids.append(str(i) + '_TEST')  

            all_ids= new_ids
        
        all_ids = np.array(all_ids)
        np.save(output_path + "item_ids_" + clip_version +".npy", all_ids)
        
def find_images(folder):
    images = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                images.append(os.path.join(root, file))
    return images

def check_paths(img_id, img_paths):
    for path in img_paths:

        if img_id in path:            
            return path

def unpack_data(input_df, img_paths):

    keep_data = []

    for i, row in input_df.iterrows():
        current_row = row.copy()
        image_ids = row.image_id.split(',')

        if len(image_ids) >= 2:
            
            for img_id in image_ids:
                current_row['image_id'] = img_id
                current_row["image_path"] = check_paths(img_id, img_paths)
                
                keep_data.append(current_row.to_dict())
                
        else: 
            
            img_id = current_row.image_id
            current_row["image_path"] = check_paths(img_id, img_paths)            
            keep_data.append(current_row.to_dict())
            
    return pd.DataFrame(keep_data)

def load_twitter():

    train_a = pd.read_csv('Twitter/image-verification-corpus-master/mediaeval2015/devset/tweets.txt', sep="\t")
    train_b = pd.read_csv('Twitter/image-verification-corpus-master/mediaeval2015/testset/tweets.txt', sep="\t")
    train_c = pd.read_csv ("Twitter/image-verification-corpus-master/mediaeval2016/devset/posts.txt",  sep="\t")

    train_a.rename({'tweetId': 'id', 'imageId(s)': 'image_id', 'label': 'falsified', 'tweetText': 'caption'}, axis=1, inplace=True)
    train_b.rename({'tweetId': 'id', 'imageId(s)': 'image_id', 'label': 'falsified', 'tweetText': 'caption'}, axis=1, inplace=True)
    train_c.rename({'post_id': 'id', 'image_id(s)': 'image_id', 'label': 'falsified', 'post_text': 'caption'}, axis=1, inplace=True)

    test_df = pd.read_csv ("Twitter/image-verification-corpus-master/mediaeval2016/testset/posts_groundtruth.txt",  sep="\t")
    test_df = test_df.rename({'post_id': 'id', 'imageId(s)': 'image_id', 'label': 'falsified', 'post_text': 'caption'}, axis=1)

    train_df = pd.concat([train_a, train_b, train_c])
    
    mediaeval2015_a = find_images('Twitter/image-verification-corpus-master/mediaeval2015/devset/Medieval2015_DevSet_Images')
    mediaeval2015_b = find_images('Twitter/image-verification-corpus-master/mediaeval2015/testset/TestSetImages')
    test_paths = find_images('Twitter/image-verification-corpus-master/mediaeval2016/testset/Mediaeval2016_TestSet_Images')

    dev_paths = mediaeval2015_a + mediaeval2015_b
    all_paths = dev_paths + test_paths
      
    train_df = unpack_data(train_df, all_paths)
    test_df = unpack_data(test_df, all_paths)      

    train_df = train_df[~train_df.image_path.isna()]
    train_df = train_df[train_df.falsified != 'humor']
    test_df = test_df[~test_df.image_path.isna()]
    
    return train_df, test_df

def extract_CLIP_twitter(output_path = 'Twitter/', choose_clip_version = "ViT-L/14", choose_gpu = 0):
    
    print("Load Twitter data")
    train_df, test_df = load_twitter()
    all_data = pd.concat([train_df, test_df])
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(choose_gpu)
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    print("Load CLIP")
    model, preprocess = clip.load(choose_clip_version, device=device)    
    model.eval()

    tweets = all_data.copy()
    images = tweets.drop_duplicates('image_id')

    print("Extract CLIP features from the images")
    images_ids = []
    all_visual_features = []
    with torch.no_grad():

        for (row) in tqdm(images.itertuples(), total=images.shape[0]):     
            path =  row.image_path                
            image = preprocess(Image.open(path)).unsqueeze(0).to(device)
            image_features = model.encode_image(image)

            all_visual_features.append(image_features.cpu().detach().numpy()[0])
            images_ids.append(row.image_id)
            
    print("Save: ", output_path)

    print("Save visual features")
    clip_version = choose_clip_version.replace('-', '').replace('/', '')

    all_visual_features = np.array(all_visual_features).reshape(len(all_visual_features),-1)
    np.save(output_path + "clip_image_embeddings_" + clip_version + ".npy", all_visual_features) 
    all_ids = np.array(images_ids)
    np.save(output_path + "image_item_ids_" + clip_version +".npy", all_ids)

    print("Extract CLIP features from the tweets")
    text_ids = []
    all_text_features = []
    for (row) in tqdm(tweets.itertuples(), total=tweets.shape[0]):     

        _, text = preprocess_input(None, row.caption, preprocess, device)

        if text != None: 
            text_features = model.encode_text(text)        
            all_text_features.append(text_features.cpu().detach().numpy()[0])

            text_ids.append(row.id)

    print("Save textual features")
    all_text_features = np.array(all_text_features).reshape(len(all_text_features),-1)  
    np.save(output_path + "clip_text_embeddings_" + clip_version + ".npy", all_text_features)

    all_ids = np.array(text_ids)
    np.save(output_path + "text_item_ids_" + clip_version +".npy", all_ids)

    # Keep only data that have features
    train_df = train_df[train_df.id.isin(text_ids)]
    test_df = test_df[test_df.id.isin(text_ids)]

    # Save final dataframes as .csv
    train_df.to_csv('Twitter/train.csv')
    test_df.to_csv('Twitter/test.csv')


