import os
import json
import time
import torch 
import numpy as np
import pandas as pd
from sklearn import metrics
from torch.utils.data import DataLoader
from imblearn.under_sampling import RandomUnderSampler 

class DatasetIterator(torch.utils.data.Dataset):
    def __init__(
        self,
        input_data,
        visual_features,
        textual_features
    ):
        self.input_data = input_data
        self.visual_features = visual_features
        self.textual_features = textual_features

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, idx):
        current = self.input_data.iloc[idx]
        
        img = self.visual_features[current.image_id].values
        txt = self.textual_features[current.id].values
        label = float(current.falsified)
        
        return img, txt, label
    
def prepare_dataloader(image_embeddings, text_embeddings, input_data, batch_size, num_workers, shuffle):
    dg = DatasetIterator(
        input_data,
        visual_features=image_embeddings,
        textual_features=text_embeddings
    )

    dataloader = DataLoader(
        dg,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return dataloader


def binary_acc(y_pred, y_test):
    
    y_pred = torch.sigmoid(y_pred)
    y_pred_tag = torch.round(y_pred)

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc

def topsis(xM, wV=None):
    m, n = xM.shape

    if wV is None:
        wV = np.ones((1, n)) / n
    else:
        wV = wV / np.sum(wV)

    normal = np.sqrt(np.sum(xM**2, axis=0))

    rM = xM / normal
    tM = rM * wV
    twV = np.max(tM, axis=0)
    tbV = np.min(tM, axis=0)
    dwV = np.sqrt(np.sum((tM - twV) ** 2, axis=1))
    dbV = np.sqrt(np.sum((tM - tbV) ** 2, axis=1))
    swV = dwV / (dwV + dbV)

    arg_sw = np.argsort(swV)[::-1]

    r_sw = swV[arg_sw]

    return np.argsort(swV)[::-1]

def choose_best_model(input_df, metrics, epsilon=1e-6):

    X0 = input_df.copy()
    X0 = X0.reset_index(drop=True)
    X1 = X0[metrics]
    X1 = X1.reset_index(drop=True)
    
    # Stop if the scores are identical in all consecutive epochs
    X1[:-1] = X1[:-1] + epsilon

    if "Accuracy" in metrics:
        X1["Accuracy"] = 1 - X1["Accuracy"]    

    if "Precision" in metrics:
        X1["Precision"] = 1 - X1["Precision"]    

    if "Recall" in metrics:
        X1["Recall"] = 1 - X1["Recall"]          
        
    if "AUC" in metrics:
        X1["AUC"] = 1 - X1["AUC"]
        
    if "F1" in metrics:
        X1["F1"] = 1 - X1["F1"]

    if "Pristine" in metrics:
        X1["Pristine"] = 1 - X1["Pristine"]
        
    if "Falsified" in metrics:
        X1["Falsified"] = 1 - X1["Falsified"]
        
    X_np = X1.to_numpy()
    best_results = topsis(X_np)
    top_K_results = best_results[:1]
    return X0.iloc[top_K_results]

def save_results_csv(output_folder_, output_file_, model_performance_):
    print("Save Results ", end=" ... ")
    exp_results_pd = pd.DataFrame(pd.Series(model_performance_)).transpose()
    if not os.path.isfile(output_folder_ + "/" + output_file_ + ".csv"):
        exp_results_pd.to_csv(
            output_folder_ + "/" + output_file_ + ".csv",
            header=True,
            index=False,
            columns=list(model_performance_.keys()),
        )
    else:
        exp_results_pd.to_csv(
            output_folder_ + "/" + output_file_ + ".csv",
            mode="a",
            header=False,
            index=False,
            columns=list(model_performance_.keys()),
        )
    print("Done\n")

    
def down_sample(input_data):
    
    rus = RandomUnderSampler(random_state=0)
    X, y = rus.fit_resample(input_data[['id', 'image_id']], input_data['falsified'])
    X['falsified'] = y
    
    return X


def early_stop(has_not_improved_for, model, optimizer, history, current_epoch, PATH, metrics_list):

    best_index = choose_best_model(
        pd.DataFrame(history), metrics=metrics_list
    ).index[0]
    
    if not os.path.isdir(PATH.split('/')[0]):
        os.mkdir(PATH.split('/')[0])

    if current_epoch == best_index:
        
        print("Checkpoint!\n")
        torch.save(
            {
                "epoch": current_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            PATH,
        )

        has_not_improved_for = 0
    else:
        
        print("DID NOT CHECKPOINT!\n")
        has_not_improved_for += 1
            
    return has_not_improved_for


def train_step(model, input_dataloader, current_epoch, optimizer, criterion, device, batches_per_epoch, use_multiclass=False):
    epoch_start_time = time.time()

    running_loss = 0.0
    model.train()

    for i, data in enumerate(input_dataloader, 0):

        images = data[0].to(device, non_blocking=True)
        texts = data[1].to(device, non_blocking=True)

        if use_multiclass:
            labels = torch.nn.functional.one_hot(data[2].long(), num_classes=3).float().to(device, non_blocking=True)
        else:
            labels = data[2].to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images, texts)

        loss = criterion(
            outputs, labels if use_multiclass else labels.unsqueeze(1)
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        print(
            f"[Epoch:{current_epoch + 1}, Batch:{i + 1:5d}/{batches_per_epoch}]. Passed time: {round((time.time() - epoch_start_time) / 60, 1)} minutes. loss: {running_loss / (i+1):.3f}",
            end="\r",
        )          
        

def eval_step(model, input_dataloader, current_epoch, device, use_multiclass=False, return_results=True):

    if current_epoch >= 0:
        print("\nEvaluation:", end=" -> ")
    else:
        print("\nFinal evaluation on the TESTING set", end=" -> ")

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for i, data in enumerate(input_dataloader, 0):

                images = data[0].to(device, non_blocking=True)
                texts = data[1].to(device, non_blocking=True)
                labels = data[2].to(device, non_blocking=True)

                predictions = model(images, texts)
                y_pred.extend(predictions.cpu().detach().numpy())
                y_true.extend(labels.cpu().detach().numpy())

    y_pred = np.vstack(y_pred)

    if use_multiclass:
        y_true = np.vstack(y_true)
        y_true = y_true.flatten()
        y_pred_softmax = torch.log_softmax(torch.Tensor(y_pred), dim = 1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
        y_pred = y_pred_tags.numpy() 
        
        if not return_results:
            return y_true, y_pred
        
        acc = metrics.accuracy_score(y_true, y_pred)    
        prec = metrics.precision_score(y_true, y_pred, average='macro')
        recall = metrics.recall_score(y_true, y_pred, average='macro') 
        f1 = metrics.f1_score(y_true, y_pred, average='macro')

        results = {
            "epoch": current_epoch,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
        }
        print(results)

    else:
        y_pred = 1/(1 + np.exp(-y_pred))
        y_true = np.array(y_true).reshape(-1,1)
        
        if not return_results:
            return y_true, y_pred

        auc = metrics.roc_auc_score(y_true, y_pred)
        y_pred = np.round(y_pred)        
        acc = metrics.accuracy_score(y_true, y_pred)    
        prec = metrics.precision_score(y_true, y_pred)
        recall = metrics.recall_score(y_true, y_pred) 
        f1 = metrics.f1_score(y_true, y_pred)

        cm = metrics.confusion_matrix(y_true, y_pred, normalize="true").diagonal()

        results = {
            "epoch": current_epoch,
            "Accuracy": round(acc, 4),
            "AUC": round(auc, 4),
            "Precision": round(prec, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
            'Pristine': round(cm[0], 4),
            'Falsified': round(cm[1], 4)
        }
        print(results)
    
    return results


def eval_cosmos(model, clip_version, device, batch_size, num_workers, use_multiclass=False):
    data = []

    for line in open('COSMOS/cosmos_anns/test_data.json', 'r'):
        data.append(json.loads(line))

    cosmos_test = pd.DataFrame(data)
    cosmos_text_embeddings = np.load("COSMOS/COSMOS_clip_text_embeddings_test_" + clip_version + ".npy").astype('float32')
    cosmos_image_embeddings = np.load("COSMOS/COSMOS_clip_image_embeddings_test_" + clip_version + ".npy").astype('float32')

    # Alter COSMOS to be similar to VisualNews in order to re-use the same evaluation functions
    cosmos_test.index.name = 'image_id'
    cosmos_test = cosmos_test.reset_index()
    cosmos_test['id'] = cosmos_test['image_id']
    cosmos_test.rename({'context_label': 'falsified'}, axis=1, inplace=True)

    cosmos_image_embeddings = pd.DataFrame(cosmos_image_embeddings, index=cosmos_test.id.values).T
    cosmos_text_embeddings = pd.DataFrame(cosmos_text_embeddings, index=cosmos_test.id.values).T

    cosmos_dataloader = prepare_dataloader(cosmos_image_embeddings, cosmos_text_embeddings, cosmos_test, batch_size, num_workers, False)
    
    if use_multiclass:
        y_true, y_pred = eval_step(model, cosmos_dataloader, -1, device, use_multiclass=True, return_results=False)
        y_pred[np.where(y_pred > 0)] = 1
        
        acc = metrics.accuracy_score(y_true, y_pred)    
        prec = metrics.precision_score(y_true, y_pred)
        recall = metrics.recall_score(y_true, y_pred) 
        f1 = metrics.f1_score(y_true, y_pred)

        cm = metrics.confusion_matrix(y_true, y_pred, normalize="true").diagonal()

        cosmos_results = {
            "epoch": -1,
            "Accuracy": round(acc, 4),
            "AUC": 0,
            "Precision": round(prec, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
            'Pristine': round(cm[0], 4),
            'Falsified': round(cm[1], 4)
        }
        
        print(cosmos_results)
        
    else:
        cosmos_results = eval_step(model, cosmos_dataloader, -1, device)
    
    return cosmos_results

def check_C(C, pos):
    
    if C == 0:
        return np.zeros(pos.shape[0])    
    else: 
        return np.ones(pos.shape[0])
        
        
def sensitivity_per_class(y_true, y_pred, C):
    
    pos = np.where(y_true == C)[0]
    y_true = y_true[pos]
    y_pred = y_pred[pos]
    
    if C == 2:
        y_true = np.ones(y_true.shape[0]).reshape(-1, 1)
    
    return round((y_pred == y_true).sum() / y_true.shape[0], 4)

def accuracy_CvC(y_true, y_pred, Ca, Cb):
    pos_a, _ = np.where(y_true == Ca)
    pos_b, _ = np.where(y_true == Cb)

    y_pred_a = y_pred[pos_a].flatten()
    y_pred_b = y_pred[pos_b].flatten()   
    
    y_true_a = check_C(Ca, pos_a)
    y_true_b = check_C(Cb, pos_b)
    
    y_pred_avb = np.concatenate([y_pred_a, y_pred_b])
    y_true_avb = np.concatenate([y_true_a, y_true_b])
    
    return round(metrics.accuracy_score(y_true_avb, y_pred_avb), 4)

def eval_figments(model, clip_version, device, batch_size, num_workers, use_multiclass=False, label_map={'true': 0, 'miscaptioned': 1, 'out-of-context': 2}):
    
    figments_test = pd.read_csv('FIGMENTS/FIGMENTS.csv', index_col=0)
    figments_test = figments_test.reset_index().rename({'index': 'id', 'label': 'falsified'}, axis=1)
    figments_test['image_id'] = figments_test['id']
    
    figments_text_embeddings = np.load("FIGMENTS/FIGMENTS_clip_text_embeddings_" + clip_version + ".npy").astype('float32')
    figments_image_embeddings = np.load("FIGMENTS/FIGMENTS_clip_image_embeddings_" + clip_version + ".npy").astype('float32')

    figments_image_embeddings = pd.DataFrame(figments_image_embeddings, index=figments_test.id.values).T
    figments_text_embeddings = pd.DataFrame(figments_text_embeddings, index=figments_test.id.values).T
    
    figments_test.falsified.replace(label_map, inplace=True)
    figments_dataloader = prepare_dataloader(figments_image_embeddings, figments_text_embeddings, figments_test, batch_size, num_workers, False)
    
    y_true, y_pred = eval_step(model, figments_dataloader, -1, device, use_multiclass=use_multiclass, return_results=False)
        
    if use_multiclass:
        acc = metrics.accuracy_score(y_true, y_pred)   
        matrix = metrics.confusion_matrix(y_true, y_pred)
        cm_results = matrix.diagonal() / matrix.sum(axis=1)

        true_ = cm_results[0]
        miscaptioned_ = cm_results[1]
        out_of_context = cm_results[2]

        figments_results = {
            "epoch": -1,
            "Accuracy": round(acc, 4),
            'True': round(cm_results[0], 4),
            'Miscaptioned': round(cm_results[1], 4),
            'Out-Of-Context': round(cm_results[2], 4)        
        }
        
    else:
        y_pred = y_pred.round()

        figments_results = {}
        
        figments_results['epoch'] = -1
        
        figments_results['True'] = sensitivity_per_class(y_true, y_pred, 0)
        figments_results['Miscaptioned'] = sensitivity_per_class(y_true, y_pred, 1)
        figments_results['Out-Of-Context'] = sensitivity_per_class(y_true, y_pred, 2)
        
        figments_results['true_v_miscaptioned'] = accuracy_CvC(y_true, y_pred, 0, 1)
        figments_results['true_v_ooc'] = accuracy_CvC(y_true, y_pred, 0, 2)
        figments_results['miscaptioned_v_ooc'] = accuracy_CvC(y_true, y_pred, 1, 2)
        
        y_true_all = y_true.copy()
        y_true_all[np.where(y_true_all == 2)[0]] = 1

        figments_results['accuracy'] = round(metrics.accuracy_score(y_true_all, y_pred), 4)
        figments_results['balanced_accuracy'] = round(metrics.balanced_accuracy_score(y_true_all, y_pred), 4)
    
    print(figments_results)
    return figments_results


def load_data(choose_dataset, choose_columns=['id', 'image_id', 'falsified']):

    print("Load data for:", choose_dataset)
    
    if choose_dataset == "news_clippings":
        train_data = json.load(open("news_clippings/data/news_clippings/data/merged_balanced/train.json"))
        valid_data = json.load(open("news_clippings/data/news_clippings/data/merged_balanced/val.json"))
        test_data = json.load(open("news_clippings/data/news_clippings/data/merged_balanced/test.json"))

        train_data = pd.DataFrame(train_data["annotations"])
        valid_data = pd.DataFrame(valid_data["annotations"])
        test_data = pd.DataFrame(test_data["annotations"])

        train_data = train_data.sample(frac=1, random_state=0)
        
    elif choose_dataset == "news_clippings_txt2img":
        
        train_data = json.load(open("news_clippings/data/news_clippings/data/semantics_clip_text_image/train.json"))
        valid_data = json.load(open("news_clippings/data/news_clippings/data/semantics_clip_text_image/val.json"))
        test_data = json.load(open("news_clippings/data/news_clippings/data/semantics_clip_text_image/test.json"))

        train_data = pd.DataFrame(train_data["annotations"])
        valid_data = pd.DataFrame(valid_data["annotations"])
        test_data = pd.DataFrame(test_data["annotations"])

        train_data = train_data.sample(frac=1, random_state=0)

    elif choose_dataset == "news_clippings_txt2txt":
        
        train_data = json.load(open("news_clippings/data/news_clippings/data/semantics_clip_text_text/train.json"))
        valid_data = json.load(open("news_clippings/data/news_clippings/data/semantics_clip_text_text/val.json"))
        test_data = json.load(open("news_clippings/data/news_clippings/data/semantics_clip_text_text/test.json"))

        train_data = pd.DataFrame(train_data["annotations"])
        valid_data = pd.DataFrame(valid_data["annotations"])
        test_data = pd.DataFrame(test_data["annotations"])

        train_data = train_data.sample(frac=1, random_state=0)        
    
    elif choose_dataset == "random_sampling":
        train_data = pd.read_csv('VisualNews/train_random_sample.csv', index_col=0)
        valid_data = pd.read_csv('VisualNews/valid_random_sample.csv', index_col=0)
        test_data = pd.read_csv('VisualNews/test_random_sample.csv', index_col=0)     

    elif choose_dataset == "random_sampling_topic":
        train_data = pd.read_csv('VisualNews/train_random_sample_topic.csv', index_col=0)
        valid_data = pd.read_csv('VisualNews/valid_random_sample_topic.csv', index_col=0)
        test_data = pd.read_csv('VisualNews/test_random_sample_topic.csv', index_col=0)     

    elif choose_dataset == "random_sampling_topic_image":
        train_data = pd.read_csv('VisualNews/train_random_sample_topic_image.csv', index_col=0)
        valid_data = pd.read_csv('VisualNews/valid_random_sample_topic_image.csv', index_col=0)
        test_data = pd.read_csv('VisualNews/test_random_sample_topic_image.csv', index_col=0) 
        
    elif choose_dataset == "random_sampling_topic_text":
        train_data = pd.read_csv('VisualNews/train_random_sample_topic_text.csv', index_col=0)
        valid_data = pd.read_csv('VisualNews/valid_random_sample_topic_text.csv', index_col=0)
        test_data = pd.read_csv('VisualNews/test_random_sample_topic_text.csv', index_col=0)         
                
    elif choose_dataset == "meir":
        train_data = pd.read_csv('MEIR/train_meir.csv', index_col=0)
        valid_data = pd.read_csv('MEIR/valid_meir.csv', index_col=0)
        test_data = pd.read_csv('MEIR/test_meir.csv', index_col=0)   
        
    elif "Misalign" in choose_dataset:
        train_data = pd.read_csv('VisualNews/train_Misalign.csv', index_col=0)
        valid_data = pd.read_csv('VisualNews/valid_Misalign.csv', index_col=0)        
        test_data = pd.read_csv('VisualNews/test_Misalign.csv', index_col=0)
        
        if choose_dataset == 'Misalign_D':
            train_data = train_data.sample(frac=1).drop_duplicates('id')
            valid_data = valid_data.sample(frac=1).drop_duplicates('id')
            test_data = test_data.sample(frac=1).drop_duplicates('id')
        
    elif choose_dataset == "clip_based_sampling_topic":
        train_data = pd.read_csv('VisualNews/train_clip_based_sampling_topic.csv', index_col=0)
        valid_data = pd.read_csv('VisualNews/valid_clip_based_sampling_topic.csv', index_col=0)        
        test_data = pd.read_csv('VisualNews/test_clip_based_sampling_topic.csv', index_col=0)
        
    elif choose_dataset == "clip_based_sampling_topic_image":
        train_data = pd.read_csv('VisualNews/train_clip_based_sampling_topic_image.csv', index_col=0)
        valid_data = pd.read_csv('VisualNews/valid_clip_based_sampling_topic_image.csv', index_col=0)        
        test_data = pd.read_csv('VisualNews/test_clip_based_sampling_topic_image.csv', index_col=0)
        
    elif choose_dataset == "clip_based_sampling_topic_text":
        train_data = pd.read_csv('VisualNews/train_clip_based_sampling_topic_text.csv', index_col=0)
        valid_data = pd.read_csv('VisualNews/valid_clip_based_sampling_topic_text.csv', index_col=0)        
        test_data = pd.read_csv('VisualNews/test_clip_based_sampling_topic_text.csv', index_col=0)      
        
    elif choose_dataset == "EntitySwaps_random_topic":
        train_data = pd.read_csv('VisualNews/train_entity_swap_topic.csv', index_col=0)
        valid_data = pd.read_csv('VisualNews/valid_entity_swap_topic.csv', index_col=0)        
        test_data = pd.read_csv('VisualNews/test_entity_swap_topic.csv', index_col=0)   
        
    elif choose_dataset == "EntitySwaps_CLIP_topic":
        train_data = pd.read_csv('VisualNews/train_entity_swap_topic_CLIP.csv', index_col=0)
        valid_data = pd.read_csv('VisualNews/valid_entity_swap_topic_CLIP.csv', index_col=0)        
        test_data = pd.read_csv('VisualNews/test_entity_swap_topic_CLIP.csv', index_col=0)          
    
    elif choose_dataset == "EntitySwaps_CLIP_topic_bytext":
        train_data = pd.read_csv('VisualNews/train_entity_swap_topic_CLIP_text.csv', index_col=0)
        valid_data = pd.read_csv('VisualNews/valid_entity_swap_topic_CLIP_text.csv', index_col=0)        
        test_data = pd.read_csv('VisualNews/test_entity_swap_topic_CLIP_text.csv', index_col=0)          
        
    elif choose_dataset == "EntitySwaps_CLIP_topic_byimage":
        train_data = pd.read_csv('VisualNews/train_entity_swap_topic_CLIP_image.csv', index_col=0)
        valid_data = pd.read_csv('VisualNews/valid_entity_swap_topic_CLIP_image.csv', index_col=0)        
        test_data = pd.read_csv('VisualNews/test_entity_swap_topic_CLIP_image.csv', index_col=0)  
        
    elif choose_dataset == "fakeddit_original":
        train_data = pd.read_csv('Fakeddit/all_samples/all_train.tsv', sep='\t')
        valid_data = pd.read_csv('Fakeddit/all_samples/all_validate.tsv', sep='\t')
        test_data = pd.read_csv('Fakeddit/all_samples/all_test_public.tsv', sep='\t') 
        
        train_data = train_data[~train_data.image_url.isna()]
        train_data = train_data[~train_data.clean_title.isna()]

        valid_data = valid_data[~valid_data.image_url.isna()]
        valid_data = valid_data[~valid_data.clean_title.isna()]

        test_data = test_data[~test_data.image_url.isna()]
        test_data = test_data[~test_data.clean_title.isna()]
        
        train_data['image_id'] = train_data["id"]
        valid_data['image_id'] = valid_data["id"]
        test_data['image_id'] = test_data["id"]  
        
        train_data["falsified"] = train_data["2_way_label"]
        valid_data["falsified"] = valid_data["2_way_label"]
        test_data["falsified"] = test_data["2_way_label"]
        
        id_list = np.load('Fakeddit/fd_original_clip_item_ids_ViTL14.npy')
        
        train_data = train_data[train_data.id.isin(id_list)]
        valid_data = valid_data[valid_data.id.isin(id_list)]
        test_data = test_data[test_data.id.isin(id_list)]        
        
        train_data.falsified.replace({0:'1', 1:'0'}, inplace=True)
        valid_data.falsified.replace({0:'1', 1:'0'}, inplace=True)        
        test_data.falsified.replace({0:'1', 1:'0'}, inplace=True)        
        
    elif "Twitter" in choose_dataset:
        train_data = pd.read_csv('Twitter/train.csv', index_col=0)
        test_data = pd.read_csv('Twitter/test.csv', index_col=0)
        
        train_data.falsified = train_data.falsified.replace({'fake': 1, 'real': 0})
        test_data.falsified = test_data.falsified.replace({'fake': 1, 'real': 0})        
        
        if choose_dataset == "Twitter_comparable":
            valid_data = test_data.copy()
            
        elif choose_dataset == "Twitter_corrected":
            valid_data = train_data.sample(frac=0.1, random_state=0)
            train_data = train_data[~train_data.id.isin(valid_data.id.tolist())]
                      
    train_data.id = train_data.id.astype('str')
    valid_data.id = valid_data.id.astype('str')
    test_data.id = test_data.id.astype('str')
    
    train_data.image_id = train_data.image_id.astype('str')
    valid_data.image_id = valid_data.image_id.astype('str')
    test_data.image_id = test_data.image_id.astype('str')
    
    train_data.reset_index(drop=True, inplace=True)
    valid_data.reset_index(drop=True, inplace=True)    
    test_data.reset_index(drop=True, inplace=True)
    
    return train_data[choose_columns], valid_data[choose_columns], test_data[choose_columns] 


def load_ensemble_data(dataset_method, use_multiclass, choose_columns=['id', 'image_id', 'falsified']):

    dataset_list = dataset_method.split('X')

    a_train_data, a_valid_data, a_test_data = load_data(dataset_method.split('X')[0])        
    b_train_data, b_valid_data, b_test_data = load_data(dataset_method.split('X')[-1])

    a_train_data.loc[a_train_data.falsified == True, 'falsified'] = 'tempered_occ'
    a_valid_data.loc[a_valid_data.falsified == True, 'falsified'] = 'tempered_occ'
    a_test_data.loc[a_test_data.falsified == True, 'falsified'] = 'tempered_occ'

    b_train_data.loc[b_train_data.falsified == True, 'falsified'] = 'untempered_occ'
    b_valid_data.loc[b_valid_data.falsified == True, 'falsified'] = 'untempered_occ'
    b_test_data.loc[b_test_data.falsified == True, 'falsified'] = 'untempered_occ'

    if len(dataset_list) == 3:
        a2_train_data, a2_valid_data, a2_test_data = load_data(dataset_method.split('X')[1])   
        a2_train_data.loc[a2_train_data.falsified == True, 'falsified'] = 'tempered_occ'
        a2_valid_data.loc[a2_valid_data.falsified == True, 'falsified'] = 'tempered_occ'
        a2_test_data.loc[a2_test_data.falsified == True, 'falsified'] = 'tempered_occ'

    elif len(dataset_list) > 3:
        raise BaseException("Error, cannot combine more than 3 datasets.")

    if len(dataset_list) == 3:
        train_data = pd.concat([a_train_data, a2_train_data, b_train_data])
        valid_data = pd.concat([a_valid_data, a2_valid_data, b_valid_data])
        test_data = pd.concat([a_test_data, a2_test_data, b_test_data])

    else:
        train_data = pd.concat([a_train_data, b_train_data])
        valid_data = pd.concat([a_valid_data, b_valid_data])
        test_data = pd.concat([a_test_data, b_test_data])        

    if use_multiclass:
        label_map={'true': 0, 'tempered_occ': 1, 'untempered_occ': 2}
    else:
        label_map={'true': 0, 'tempered_occ': 1, 'untempered_occ': 1}        

    train_data = train_data.drop_duplicates(['id', 'image_id', 'falsified'], keep='first')
    valid_data = valid_data.drop_duplicates(['id', 'image_id', 'falsified'], keep='first')
    test_data = test_data.drop_duplicates(['id', 'image_id', 'falsified'], keep='first')

    train_data.loc[train_data.falsified == False, 'falsified'] = 'true'
    valid_data.loc[valid_data.falsified == False, 'falsified'] = 'true'
    test_data.loc[test_data.falsified == False, 'falsified'] = 'true'

    train_data = train_data.sample(frac=1)

    train_data.reset_index(drop=True, inplace=True)
    valid_data.reset_index(drop=True, inplace=True)    
    test_data.reset_index(drop=True, inplace=True)

    train_data.falsified.replace(label_map, inplace=True)
    valid_data.falsified.replace(label_map, inplace=True)
    test_data.falsified.replace(label_map, inplace=True)

    print(train_data.falsified.value_counts())

    return train_data[choose_columns], valid_data[choose_columns], test_data[choose_columns]

def load_features(input_parameters):
    
    clip_version = input_parameters["CLIP_VERSION"].replace("-", "").replace("/", "")
    
    print("Load features for:", input_parameters["CHOOSE_DATASET"])
    
    if input_parameters["CHOOSE_DATASET"] == "meir":
        clip_text_embeddings = np.load(
            "MEIR/MEIR_clip_text_embeddings_" + clip_version + ".npy"
        ).astype("float32")
        clip_image_embeddings = np.load(
            "MEIR/MEIR_clip_image_embeddings_" + clip_version + ".npy"
        ).astype("float32")

        clip_image_embeddings = pd.DataFrame(clip_image_embeddings).T
        clip_text_embeddings = pd.DataFrame(clip_text_embeddings).T
        
    elif 'Twitter' in input_parameters["CHOOSE_DATASET"]:
        
        clip_image_embeddings = np.load(
            "Twitter/clip_image_embeddings_" + clip_version + ".npy"
        ).astype("float32")

        clip_text_embeddings = np.load(
                "Twitter/clip_text_embeddings_" + clip_version + ".npy"
            ).astype("float32")

        text_ids = np.load("Twitter/text_item_ids_" + clip_version + ".npy")
        image_ids = np.load("Twitter/image_item_ids_" + clip_version + ".npy")

        clip_image_embeddings = pd.DataFrame(clip_image_embeddings, index=image_ids).T
        clip_text_embeddings = pd.DataFrame(clip_text_embeddings, index=text_ids).T
        clip_text_embeddings = clip_text_embeddings.loc[:,~clip_text_embeddings.columns.duplicated()].copy()
        
    elif input_parameters["CHOOSE_DATASET"] == "fakeddit_original":
        
        clip_text_embeddings = np.load(
            "Fakeddit/fd_original_clip_text_embeddings_" + clip_version + ".npy"
        ).astype("float32")
        
        clip_image_embeddings = np.load(
            "Fakeddit/fd_original_clip_image_embeddings_" + clip_version + ".npy"
        ).astype("float32")
        
        item_ids = np.load("Fakeddit/fd_original_clip_item_ids_" + clip_version + ".npy")

        clip_image_embeddings = pd.DataFrame(clip_image_embeddings, index=item_ids).T
        clip_text_embeddings = pd.DataFrame(clip_text_embeddings, index=item_ids).T    
        

    else:
        print("VisualNews features")
        
        clip_image_embeddings = np.load(
            "VisualNews/clip_image_embeddings_" + clip_version + ".npy"
        ).astype("float32")

        clip_text_embeddings = np.load(
            "VisualNews/clip_text_embeddings_" + clip_version + ".npy"
        ).astype("float32")

        item_ids = np.load("VisualNews/item_ids_" + clip_version + ".npy")

        clip_image_embeddings = pd.DataFrame(clip_image_embeddings, index=item_ids).T
        clip_text_embeddings = pd.DataFrame(clip_text_embeddings, index=item_ids).T

        if 'Misalign' in input_parameters["CHOOSE_DATASET"]:
            
            print("Misalign features")
    
            all_misalign_features = np.load("VisualNews/MISALIGN_clip_text_embeddings_" + clip_version + ".npy").astype('float32')
            all_idx = np.load("VisualNews/MISALIGN_item_ids_" + clip_version + ".npy")
            all_misalign_features = pd.DataFrame(all_misalign_features.T, columns=all_idx)
    
            clip_text_embeddings = pd.concat([clip_text_embeddings, all_misalign_features], axis=1)
            
        if 'EntitySwaps_random_topic' in input_parameters["CHOOSE_DATASET"]:
            
            NES_text_features = np.load("VisualNews/EntitySwaps_topic_random_text_embeddings_" + clip_version +".npy").astype("float32")
            NES_ids = np.load("VisualNews/EntitySwaps_topic_random_item_ids_" + clip_version +".npy")

            NES_text_features = pd.DataFrame(NES_text_features.T, columns=NES_ids)
            clip_text_embeddings = pd.concat([clip_text_embeddings, NES_text_features], axis=1)
            
        if 'EntitySwaps_CLIP_topic' in input_parameters["CHOOSE_DATASET"]:
            
            NES_text_features = np.load("VisualNews/EntitySwaps_topic_clip_text_embeddings_" + clip_version +".npy").astype("float32")
            NES_ids = np.load("VisualNews/EntitySwaps_topic_clip_item_ids_" + clip_version +".npy")

            NES_text_features = pd.DataFrame(NES_text_features.T, columns=NES_ids)
            clip_text_embeddings = pd.concat([clip_text_embeddings, NES_text_features], axis=1)  
            
        if 'EntitySwaps_CLIP_topic_bytext' in input_parameters["CHOOSE_DATASET"]:
            
            NES_text_features = np.load("VisualNews/EntitySwaps_topic_bytext_clip_text_embeddings_" + clip_version +".npy").astype("float32")
            NES_ids = np.load("VisualNews/EntitySwaps_topic_bytext_clip_item_ids_" + clip_version +".npy")

            NES_text_features = pd.DataFrame(NES_text_features.T, columns=NES_ids)
            clip_text_embeddings = pd.concat([clip_text_embeddings, NES_text_features], axis=1)      

        if 'EntitySwaps_CLIP_topic_byimage' in input_parameters["CHOOSE_DATASET"]:
            
            NES_text_features = np.load("VisualNews/EntitySwaps_topic_byimage_clip_text_embeddings_" + clip_version +".npy").astype("float32")
            NES_ids = np.load("VisualNews/EntitySwaps_topic_byimage_clip_item_ids_" + clip_version +".npy")

            NES_text_features = pd.DataFrame(NES_text_features.T, columns=NES_ids)
            clip_text_embeddings = pd.concat([clip_text_embeddings, NES_text_features], axis=1) 
                        
    clip_image_embeddings.columns = clip_image_embeddings.columns.astype('str')
    clip_text_embeddings.columns = clip_text_embeddings.columns.astype('str')    
    return clip_image_embeddings, clip_text_embeddings