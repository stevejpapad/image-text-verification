import json
import os
import copy
import time
import torch
import spacy
import pickle
import random
import shutil
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import multiprocessing as mp
from utils import load_features

def save_image(url, name):
    
    try:
        response = requests.get(url, verify=False, timeout=20)
        
        if response.status_code == 200:

            img = Image.open(BytesIO(response.content))

            width, height = img.size

            if 2400 > width > 1200 or 2400 > height > 1200:
                img = img.resize((width//2, height//2))

            if width > 2400 or height > 2400:
                img = img.resize((width//4, height//4))
            
            if not img.mode == 'RGB':
                img = img.convert('RGB')
            
            img.save("VERITE/images/" + name + ".jpg")
            
    except Exception as e: 
        print(e, "!!!", url)
        

        
def prepare_verite(download_images=True):
    
    verite = pd.read_csv('VERITE/VERITE_articles.csv', index_col=0)

    if download_images:
        
        print("Scrape images!")
        
        directory = 'VERITE/images'
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Scrape images
        for (i, row) in tqdm(verite.iterrows(), total=verite.shape[0]):
            idx = row.id
            t_url = row.true_url
            f_url = row.false_url

            save_image(t_url, "true_"+str(idx))

            if f_url:
                save_image(f_url, "false_"+str(idx))
        
    # From: true-caption, false-caption, true-image-url, false-image-url, article-url
    # Change to -> caption, image_path, label
    
    print("Unpack dataset!")
    unpack_data = []

    for (i, row) in tqdm(verite.iterrows(), total=verite.shape[0]):

        idx = row.id
        true_caption = row.true_caption
        false_caption = row.false_caption 
        true_img_path = 'images/true_' + str(idx) + '.jpg'

        unpack_data.append({
            'caption': true_caption,
            'image_path': true_img_path,
            'label': 'true'
        })

        unpack_data.append({
            'caption': false_caption,
            'image_path': true_img_path,
            'label': 'miscaptioned'
        })  

        if row.false_url:
            false_img_path = 'images/false_' + str(idx) + '.jpg'    

            unpack_data.append({
                'caption': true_caption,
                'image_path': false_img_path,
                'label': 'out-of-context'
            })        

    verite_df = pd.DataFrame(unpack_data)
    verite_df.to_csv('VERITE/VERITE.csv')

def load_split_VisualNews(clip_version="ViT-L/14", load_features=True):
    
    print("Load VisualNews")
    data = json.load(open('/fssd4/user-data/stefpapad/MISINFO/VisualNews/origin/data.json'))  
#     data = json.load(open('VisualNews/origin/data.json'))
    vn_df = pd.DataFrame(data)
    
    vn_df = vn_df.sample(frac=1, random_state=0).reset_index(drop=True)

    vn_df["id"] = vn_df["id"].astype('str') 
    vn_df["image_id"] = vn_df["id"]
    vn_df["falsified"] = False
    vn_df["type_of_alteration"] = 'None'

    total_len = vn_df.shape[0]
    train_len = int(total_len * 0.8)
    valid_len = (total_len - train_len) // 2

    train_df = vn_df.iloc[:train_len]
    valid_df = vn_df.iloc[train_len:train_len+valid_len]
    test_df = vn_df.iloc[-valid_len-1:]

    if load_features:
        print("Load embeddings")
        clip_image_embeddings = np.load("VisualNews/clip_image_embeddings_" + clip_version + ".npy")
        clip_text_embeddings = np.load("VisualNews/clip_text_embeddings_" + clip_version + ".npy")
        item_ids = np.load("VisualNews/item_ids_" + clip_version + ".npy")
        clip_image_embeddings = pd.DataFrame(clip_image_embeddings, index=item_ids).T
        clip_text_embeddings = pd.DataFrame(clip_text_embeddings, index=item_ids).T

        clip_image_embeddings.columns = clip_image_embeddings.columns.astype('str')
        clip_text_embeddings.columns = clip_text_embeddings.columns.astype('str')   

        return train_df, valid_df, test_df, clip_image_embeddings, clip_text_embeddings
    
    else:
        return train_df, valid_df, test_df

def prepare_Misalign(CLIP_VERSION = "ViT-L/14", choose_gpu = 0):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(choose_gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)

    clip_version = CLIP_VERSION.replace("-", "").replace("/", "")
    train_df, valid_df, test_df, clip_image_embeddings, clip_text_embeddings = load_split_VisualNews(clip_version)

    def cross_modal_misalignment(sample, false_df, input_name, torch_fakeddit_clip_text_embeddings):

        choice = random.choice(['image', 'text'])

        i = sample["id"]

        if choice == 'image':
            current_item = clip_image_embeddings[i]

        elif choice == 'text':
            current_item = clip_text_embeddings[i]

        a = torch.from_numpy(current_item.values).to(device)    
        all_similarites = cos_sim(a.reshape(1, -1), torch_fakeddit_clip_text_embeddings).cpu().detach().numpy()
        most_similar_id = all_similarites.argmax()
        similarity = all_similarites[most_similar_id]
        fakeddit_item = false_df.iloc[most_similar_id]
        
        sample['id'] = str(most_similar_id) + '_' + input_name # add suffix to fakeddit items! not to be mistaken with visual news items
        sample['falsified'] = True
        sample['original_caption'] = sample["caption"]
        sample['caption'] = fakeddit_item.clean_title
        sample['similarity'] = similarity
        sample["type_of_alteration"] = choice + '|' + str(fakeddit_item["6_way_label"])
        return sample

    def Misalign(input_df, false_df, input_name, method='both'):

        fakeddit_features = np.load("VisualNews/MISALIGN_clip_text_embeddings_" + clip_version + ".npy").astype('float32')
        all_idx = np.load("VisualNews/MISALIGN_item_ids_" + clip_version + ".npy")
        fakeddit_features = pd.DataFrame(fakeddit_features.T, columns=all_idx)

        cols = [x for x in fakeddit_features.columns if input_name in x]
        fakeddit_features = fakeddit_features[cols].values.T

        fakeddit_features = torch.from_numpy(fakeddit_features).to(device)

        all_generated_items = []

        for (row) in tqdm(input_df.to_dict(orient="records"), total=input_df.shape[0]):

            generated_item = cross_modal_misalignment(row, false_df, input_name, fakeddit_features)
            all_generated_items.append(generated_item)    

        return pd.DataFrame(all_generated_items)

    def apply_Misalign(input_df, split):
        
        df = pd.read_csv('Fakeddit/all_samples/all_'+ split + '.tsv', sep='\t')
        df = df[df['2_way_label'] == 0]
        df = df[~df.clean_title.isna()].reset_index(drop=True)  

        if split == 'validate':
            split = 'valid'

        if split == 'test_public':
            split = 'test'

        generated_data = Misalign(input_df, df, split.upper())
        new_df = pd.concat([generated_data, input_df])
        new_df = new_df.sample(frac=1)

        new_df.to_csv('VisualNews/' + split +'_Misalign.csv')


    apply_Misalign(train_df, 'train')
    apply_Misalign(valid_df, 'validate')
    apply_Misalign(test_df, 'test_public')
    
    
def get_K_most_similar(by_topic = True, 
                       max_tries = 3,
                       by_modality = 'both', 
                       K_most_similar = 20, 
                       CLIP_VERSION = "ViT-L/14", 
                       choose_gpu = 0):
    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    clip_version = CLIP_VERSION.replace("-", "").replace("/", "")
    
    train_df, valid_df, test_df, clip_image_embeddings, clip_text_embeddings = load_split_VisualNews(clip_version)
    
    features_dict = {
    'id': clip_text_embeddings,
    'image_id': clip_image_embeddings
    }
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(choose_gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)
    
    all_entity_types = ['CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART']

    for input_df, file_name in [
        (valid_df, 'valid_CLIP_based_similarities'), 
        (test_df, 'test_CLIP_based_similarities'), 
        (train_df, 'train_CLIP_based_similarities')
        ]:

        start_time = time.time()

        print('*****', file_name, '*****')

        input_df = input_df.fillna(value=False).reset_index(drop=True)

        input_df["id"] = input_df["id"].astype('str')
        input_df['most_similar_by_text'] = None
        input_df['most_similar_by_image'] = None

        if by_topic:
            topic_x_id_texts = input_df.groupby('topic')['id'].apply(list)
            topic_x_id_images = input_df.groupby('topic')['image_id'].apply(list)

            val_counts = input_df.topic.value_counts()
            val_counts = val_counts[val_counts >= 2]
            input_df = input_df[input_df.topic.isin(val_counts.index.tolist())]

            topic_x_id = {'image': topic_x_id_images.to_dict(),
                          'text': topic_x_id_texts.to_dict()
                         }

        def myFunc(args):

            idx, row = args

            temp_files = {}

            for modality in ['image', 'text']:

                if modality == 'image':

                    modality_id = 'image_id'
                else:
                    modality_id = 'id'

                if by_topic:
                    candidates = topic_x_id[modality][row['topic']].copy()
                else:
                    candidates =  input_df[modality_id].copy().unique().tolist()

                current_item_id = row[modality_id]
                current_item = input_df[input_df[modality_id] == current_item_id].reset_index(drop=True)

                candidates.remove(current_item_id) 

                current_item_features = features_dict[modality_id][current_item_id]                
                candidate_features = features_dict[modality_id][candidates]

                a = torch.from_numpy(current_item_features.values).to(device)    
                b = torch.from_numpy(candidate_features.values).to(device)

                all_similarites = cos_sim(a.reshape(1, -1), b.T).cpu().detach().numpy()

                most_similar_ids = all_similarites.argsort()[::-1][:K_most_similar]

                K_most_similar_IDs = np.array(candidates)[most_similar_ids]                        
                temp_files['most_similar_by_' + modality] = '/'.join([x for x in K_most_similar_IDs])

            current_item['most_similar_by_image'] = temp_files['most_similar_by_image']
            current_item['most_similar_by_text'] = temp_files['most_similar_by_text']

            return current_item.to_dict('records')

        results = []     
        for (idx,row) in tqdm(input_df.iterrows(), total=input_df.shape[0]):
            res = myFunc((idx,row))

            results.append(res)

        most_similar_data = pd.DataFrame([x[0] for x in results if x != None])
        most_similar_data.to_csv('VisualNews/' + file_name + '.csv')
    
def get_entities(txt, nlp):
    all_entity_types = ['CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART']

    ner_dict = {}
    ner_count = {}

    for key in all_entity_types:
        ner_dict[key] = '' 
        ner_dict['count_' + key] = 0

    doc = nlp(txt)
    for ent in doc.ents:

        key = ent.label_
        current_items = ner_dict[key]
        
        if current_items != '':
            current_items = current_items + '|' + ent.text
        else:
            current_items = ent.text
        ner_dict[key] = current_items
        ner_dict['count_' + key] = len(current_items.split('|'))
        
    return ner_dict

def calc_ner(input_df, nlp):
    all_ner = []
    
    input_df.reset_index(drop=True, inplace=True)

    for (row) in tqdm(input_df.itertuples(), total=input_df.shape[0]):
        ner_dict = get_entities(row.caption, nlp)

        all_ner.append(ner_dict)
    
    all_ner = pd.DataFrame(all_ner)
    all_ner['id'] = input_df['id']
    
    input_df = pd.merge(input_df, all_ner, on='id')

    return input_df

def extract_entities():
    
    nlp = spacy.load("en_core_web_trf")
    train_df, valid_df, test_df = load_split_VisualNews(load_features=False)

    ner_train_df = calc_ner(train_df, nlp)
    ner_train_df.to_csv('VisualNews/ner_train.csv')

    ner_valid_df = calc_ner(valid_df, nlp)
    ner_train_df.to_csv('VisualNews/ner_valid.csv')

    ner_test_df = calc_ner(test_df, nlp)
    ner_train_df.to_csv('VisualNews/ner_test.csv')
    

def prepare_CLIP_NESt(num_workers = 16, by_topic = True, max_tries = 5, by_modality = 'both'):
    
    random.seed(0)
    np.random.seed(0)
    
    ner_train_df = pd.read_csv('VisualNews/ner_train.csv', index_col=0)
    ner_valid_df = pd.read_csv('VisualNews/ner_valid.csv', index_col=0)
    ner_test_df = pd.read_csv('VisualNews/ner_test.csv', index_col=0)

    train_df = pd.read_csv('VisualNews/train_CLIP_based_similarities.csv', index_col=0)[['id', 'most_similar_by_text', 'most_similar_by_image']]
    valid_df = pd.read_csv('VisualNews/valid_CLIP_based_similarities.csv', index_col=0)[['id', 'most_similar_by_text', 'most_similar_by_image']]
    test_df = pd.read_csv('VisualNews/test_CLIP_based_similarities.csv', index_col=0)[['id', 'most_similar_by_text', 'most_similar_by_image']]
    
    ner_train_df = ner_train_df.merge(train_df, on='id')
    ner_valid_df = ner_valid_df.merge(valid_df, on='id')
    ner_test_df = ner_test_df.merge(test_df, on='id')
    
    all_entity_types = ['CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART']

    for input_df, file_name in [
        (ner_train_df, 'train_entity_swap_topic_CLIP'),
        (ner_valid_df, 'valid_entity_swap_topic_CLIP'), 
        (ner_test_df, 'test_entity_swap_topic_CLIP'), 
        ]:

        start_time = time.time()

        print('*****', file_name, '*****')

        input_df = input_df.fillna(value=False).reset_index(drop=True)

        input_df["id"] = input_df["id"].astype('str')
        input_df["original_caption"] = input_df["caption"]
        input_df["num_of_alterations"] = 0
        input_df["falsified"] = False
        input_df["type_of_alteration"] = None
        input_df["altered_entities"] = None

        def myFunc(args):

            idx, row = args
            try:

                if by_modality == 'both':
                    random_type_of_alteration = random.choice(['id', 'image_id'])
                elif by_modality == 'image':
                    random_type_of_alteration = 'image_id'
                else:
                    random_type_of_alteration = 'id'  

                if random_type_of_alteration == 'image_id':
                    similar_by = 'most_similar_by_image'
                elif random_type_of_alteration == 'id':
                    similar_by = 'most_similar_by_text'

                current_item = copy.deepcopy(row)                
                most_similar_ids = current_item[similar_by].split('/')

                for tries in range(max_tries):

                    if len(most_similar_ids) >= tries:

                        similar_item_id = most_similar_ids[tries]
                        similar_item_df = input_df[input_df['id'] == similar_item_id]

                        num_of_alterations = 0
                        collect_alterations = []
                        replace_text = current_item["caption"]               

                        for entity_type in all_entity_types:

                            if current_item[entity_type] and similar_item_df[entity_type].tolist()[0]:

                                for i in range(row["count_" + entity_type]):

                                    current_entity = current_item[entity_type].split('|')[i]
                                    swap_entity = similar_item_df[entity_type].tolist()[0].split('|')[0]

                                    if swap_entity != current_entity:
                                        replace_text = replace_text.replace(current_entity, swap_entity)

                                        replaced_entity = entity_type + '/' + current_entity + '/' + swap_entity
                                        collect_alterations.append(replaced_entity)

                                        num_of_alterations += 1    

                                        if num_of_alterations > len(similar_item_df[entity_type].tolist()[0].split('|')):
                                            break

                        if num_of_alterations > 0:      

                            current_item['id'] = current_item['id'] + "_alt"
                            current_item['num_of_alterations'] = num_of_alterations
                            current_item['falsified'] = True
                            current_item['caption'] = replace_text

                            to_str = '|'.join(collect_alterations)
                            current_item["altered_entities"] = to_str
                            current_item['type_of_alteration'] = random_type_of_alteration

                            return current_item.to_dict()

            except Exception as e:
                print(e)
                return None

        with mp.Pool(processes=num_workers) as executor:
            results = executor.map(myFunc,[(idx, row) for idx,row in input_df.iterrows()])

        falsified_data = pd.DataFrame([x for x in results if x != None])
        all_data = pd.concat([input_df, falsified_data])
        all_data = all_data.sample(frac=1).reset_index(drop=True)

        all_data.to_csv('VisualNews/' + file_name + '.csv')
        
        
def random_sampling_method(input_df, by_topic=False, by_modality='both'):
    
    random_falsified_data = []

    if by_topic:
        topic_x_id = input_df.groupby('topic')['id'].apply(list)
        
        val_counts = input_df.topic.value_counts()
        val_counts = val_counts[val_counts >= 2]
        input_df = input_df[input_df.topic.isin(val_counts.index.tolist())]
    
    for (row) in tqdm(input_df.to_dict(orient="records"), total=input_df.shape[0]):
        while True:
            
            if by_modality == 'both':
                random_type_of_alteration = random.choice(['id', 'image_id'])
            elif by_modality == 'image':
                random_type_of_alteration = 'image_id'
            else:
                random_type_of_alteration = 'id'                
            
            if by_topic:
                candidates = topic_x_id[row['topic']]
                
                random_item = random.choice(candidates)
                random_item = input_df[input_df[random_type_of_alteration] == random_item]
                
            else:
                random_item = input_df.sample(1)

            if random_item[random_type_of_alteration].tolist()[0] != row[random_type_of_alteration]:
                break

        row[random_type_of_alteration] = random_item[random_type_of_alteration].tolist()[0]
        
        if random_type_of_alteration == 'id':
            row['caption'] = random_item.caption.tolist()[0]
            row['article_path'] = random_item.article_path.tolist()[0]
        elif random_type_of_alteration == 'image_id':
            row['image_path'] = random_item.image_path.tolist()[0]
        
        
        row['falsified'] = True
        row['type_of_alteration'] = "altered_" + random_type_of_alteration
        random_falsified_data.append(row)

    input_df_false = pd.DataFrame(random_falsified_data, 
                                  columns=input_df.columns)
    
    new_df = pd.concat([input_df, input_df_false])
    new_df = new_df.sample(frac=1).reset_index(drop=True)
    
    return new_df


def prepare_R_NESt():
    
    random.seed(0)
    np.random.seed(0)
        
    train_df, valid_df, test_df = load_split_VisualNews(load_features=False)
        
    new_train_df = random_sampling_method(train_df)
    new_train_df.to_csv('VisualNews/train_random_sample.csv')

    new_valid_df = random_sampling_method(valid_df)
    new_valid_df.to_csv('VisualNews/valid_random_sample.csv')

    new_test_df = random_sampling_method(test_df)
    new_test_df.to_csv('VisualNews/test_random_sample.csv')
