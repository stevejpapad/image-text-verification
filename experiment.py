import time
import random
import os, json
from PIL import Image
import pandas as pd
import pickle
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from sklearn.utils import class_weight

from model import DT_Transformer
from utils import (
    choose_best_model,
    early_stop,
    train_step,
    eval_step,
    prepare_dataloader,
    binary_acc,
    eval_cosmos,
    eval_figments,
    load_data,
    load_ensemble_data,
    save_results_csv,
    load_features,
    down_sample
)


def run_experiment(
    dataset_methods_list,
    modality_options = [
        ["images", "texts"],
        ["texts"], 
        ["images"]
    ],
    choose_CLIP_version='ViT-L/14',
    epochs=30,
    seed_options = [0],
    lr_options = [5e-5],
    batch_size_options = [512],
    tf_layers_options = [1, 4],
    tf_head_options = [2, 8],
    tf_dim_options = [128, 1024],
    use_multiclass = False, # True, False
    balancing_method = None, # None, 'downsample', 'class_weights'
    choose_gpu = 0, 
    num_workers=8,
    init_model_name = ''
):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(choose_gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    for dataset_method in dataset_methods_list:
             
        input_parameters = {
            "NUM_WORKERS": num_workers,
            "EARLY_STOP_EPOCHS": 10,  
            "CHOOSE_DATASET": dataset_method, 
            "CLIP_VERSION": choose_CLIP_version,
            "BALANCING": 'downsample' if 'Misalign_D' in dataset_method else balancing_method
        }

        if len(dataset_method.split('X')) > 1 :
            train_data, valid_data, test_data = load_ensemble_data(dataset_method=dataset_method, 
                                                                   use_multiclass=use_multiclass)
        
        else:
            train_data, valid_data, test_data = load_data(dataset_method)
        
        clip_image_embeddings, clip_text_embeddings = load_features(input_parameters)
        clip_version = input_parameters["CLIP_VERSION"].replace("-", "").replace("/", "")

        class_weights = np.array([1.0 for cl in train_data.falsified.unique()])
        print(train_data.falsified.value_counts())

        if input_parameters['BALANCING'] == 'downsample':
            train_data = down_sample(train_data)
            valid_data = down_sample(valid_data)
            test_data = down_sample(test_data)
            
            print(train_data.falsified.value_counts())

        elif input_parameters['BALANCING'] == 'class_weights':
            class_weights = class_weight.compute_class_weight(class_weight='balanced', 
                                                              y=train_data.falsified,
                                                              classes=sorted(train_data.falsified.unique()))
            print(class_weights)

        for use_features in modality_options:

            experiment = 0

            for seed in seed_options:

                if use_features == ["images", "texts"]:
                    model_name = dataset_method + '_multimodal_' + str(seed) + init_model_name

                elif use_features == ["texts"]:
                    model_name = dataset_method + '_textonly_' + str(seed) + init_model_name
                    
                else:
                    model_name = dataset_method + '_imageonly_' + str(seed) + init_model_name

                torch.manual_seed(seed)

                print("*****", seed, use_features, dataset_method, choose_CLIP_version, model_name, "*****")

                for batch_size in batch_size_options:                
                    for lr in lr_options:
                        for tf_layers in tf_layers_options:  
                            for tf_head in tf_head_options:
                                for tf_dim in tf_dim_options:

                                    experiment += 1

                                    parameters = {
                                        "LEARNING_RATE": lr,
                                        "EPOCHS": epochs, 
                                        "BATCH_SIZE": batch_size,
                                        "TF_LAYERS": tf_layers,
                                        "TF_HEAD": tf_head,
                                        "TF_DIM": tf_dim,
                                        "NUM_WORKERS": 8,
                                        "USE_FEATURES": use_features,
                                        "EARLY_STOP_EPOCHS": input_parameters["EARLY_STOP_EPOCHS"],
                                        "CHOOSE_DATASET": input_parameters["CHOOSE_DATASET"],
                                        "CLIP_VERSION": input_parameters["CLIP_VERSION"],
                                        "SEED": seed,
                                        "BALANCING": input_parameters["BALANCING"],
                                    }

                                    train_dataloader = prepare_dataloader(
                                        clip_image_embeddings,
                                        clip_text_embeddings,
                                        train_data,
                                        parameters["BATCH_SIZE"],
                                        parameters["NUM_WORKERS"],
                                        True,
                                    )
                                    valid_dataloader = prepare_dataloader(
                                        clip_image_embeddings,
                                        clip_text_embeddings,
                                        valid_data,
                                        parameters["BATCH_SIZE"],
                                        parameters["NUM_WORKERS"],
                                        False,
                                    )
                                    test_dataloader = prepare_dataloader(
                                        clip_image_embeddings,
                                        clip_text_embeddings,
                                        test_data,
                                        parameters["BATCH_SIZE"],
                                        parameters["NUM_WORKERS"],
                                        False,
                                    )

                                    print("!!!!!!!!!!!!!!!!!!!", experiment, "!!!!!!!!!!!!!!!!!!!")
                                    print("!!!!!!!!!!!!!!!!!!!", parameters, "!!!!!!!!!!!!!!!!!!!")

                                    if parameters["CLIP_VERSION"] == "ViT-L/14":
                                        emb_dim_ = 1536
                                    elif parameters["CLIP_VERSION"] == "ViT-B/32":
                                        emb_dim_ = 1024

                                    if parameters["USE_FEATURES"] == ["images", "texts"]:
                                        parameters["EMB_SIZE"] = emb_dim_
                                    else:
                                        parameters["EMB_SIZE"] = int(emb_dim_ / 2)

                                    model = DT_Transformer(
                                        device=device,
                                        tf_layers=parameters["TF_LAYERS"],
                                        tf_head=parameters["TF_HEAD"],
                                        tf_dim=parameters["TF_DIM"],
                                        emb_dim=parameters["EMB_SIZE"],
                                        use_features=parameters["USE_FEATURES"],
                                        use_multiclass=use_multiclass
                                    )

                                    model.to(device)
                                    class_weights_torch = torch.tensor(class_weights).to(device, non_blocking=True)

                                    if use_multiclass: 
                                        criterion = nn.CrossEntropyLoss(weight = class_weights_torch)
                                    else:
                                        criterion = nn.BCEWithLogitsLoss()

                                    optimizer = torch.optim.Adam(
                                        model.parameters(), lr=parameters["LEARNING_RATE"]
                                    )

                                    scheduler = torch.optim.lr_scheduler.StepLR(
                                        optimizer, step_size=30, gamma=0.1, verbose=True
                                    )

                                    batches_per_epoch = train_dataloader.__len__()

                                    history = []
                                    has_not_improved_for = 0

                                    PATH = "checkpoints_pt/model" + model_name + ".pt" # !!!!!!!!!!!!!!!!!!!!!!!!! 

                                    for epoch in range(parameters["EPOCHS"]):

                                        train_step(
                                            model,
                                            train_dataloader,
                                            epoch,
                                            optimizer,
                                            criterion,
                                            device,
                                            batches_per_epoch,
                                            use_multiclass
                                        )

                                        results = eval_step(model, valid_dataloader, epoch, device, use_multiclass=use_multiclass)
                                        history.append(results)

                                        has_not_improved_for = early_stop(
                                            has_not_improved_for,
                                            model,
                                            optimizer,
                                            history,
                                            epoch,
                                            PATH,
                                            metrics_list=["Accuracy", "F1"] if use_multiclass else ["Accuracy","AUC","Pristine","Falsified"],
                                        )

                                        if has_not_improved_for >= parameters["EARLY_STOP_EPOCHS"]:

                                            EARLY_STOP_EPOCHS = parameters["EARLY_STOP_EPOCHS"]
                                            print(
                                                f"Performance has not improved for {EARLY_STOP_EPOCHS} epochs. Stop training at epoch {epoch}!"
                                            )
                                            break

                                    print("Finished Training. Loading the best model from checkpoints.")

                                    checkpoint = torch.load(PATH)
                                    model.load_state_dict(checkpoint["model_state_dict"])
                                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                                    epoch = checkpoint["epoch"]

                                    res_val = eval_step(model, 
                                                        valid_dataloader, 
                                                        -1, 
                                                        device, 
                                                        use_multiclass=use_multiclass)                                    
                                    
                                    res_test = eval_step(model, 
                                                         test_dataloader, 
                                                         -1, 
                                                         device, 
                                                         use_multiclass=use_multiclass)

                                    cosmos_results = eval_cosmos(
                                        model,
                                        clip_version,
                                        device,
                                        parameters["BATCH_SIZE"],
                                        parameters["NUM_WORKERS"],
                                        use_multiclass=use_multiclass
                                    )

                                    figments_results = eval_figments(
                                        model,
                                        clip_version,
                                        device,
                                        parameters["BATCH_SIZE"],
                                        parameters["NUM_WORKERS"],
                                        use_multiclass=use_multiclass,
                                        label_map={'true': 0, 'miscaptioned': 1, 'out-of-context': 2},
                                    )
                                    
                                    res_val = {
                                    "valid_" + str(key.lower()): val for key, val in res_val.items()
                                    }

                                    res_test = {
                                    "test_" + str(key.lower()): val for key, val in res_test.items()
                                    }
                                    
                                    res_cosmos = {
                                    "cosmos_" + str(key): val for key, val in cosmos_results.items()
                                    }

                                    res_figments = {
                                        "figments_" + str(key): val for key, val in figments_results.items()
                                    }

                                    all_results = {**res_test, **res_val}
                                    all_results = {**res_cosmos, **all_results}
                                    all_results = {**res_figments, **all_results}
                                    all_results = {**parameters, **all_results}
                                    all_results["path"] = PATH
                                    all_results["history"] = history

                                    if not os.path.isdir("results"):
                                        os.mkdir("results")
                                    
                                    save_results_csv(
                                        "results/",
                                        "results_figments_multiclass" if use_multiclass else 'results_figments',
                                        all_results,
                                    )