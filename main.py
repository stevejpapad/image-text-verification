from experiment import run_experiment
from prepare_datasets import prepare_figments, prepare_Misalign, get_K_most_similar, extract_entities, prepare_CLIP_NESt, prepare_R_NESt
from extract_features import extract_CLIP_features, extract_CLIP_twitter

# Scrape the images of FIGMENTS and prepare the dataset
prepare_figments(download_images=True)

# Extract features with CLIP ViT-L/14 from FIGMENTS, COSMOS, Twitter, VisualNews etc
extract_CLIP_features(data_path='FIGMENTS/', output_path='FIGMENTS/FIGMENTS_')
extract_CLIP_features(data_path='COSMOS/', output_path='COSMOS/COSMOS_') 
extract_CLIP_twitter(output_path='Twitter/', choose_clip_version="ViT-L/14", choose_gpu=0)
extract_CLIP_features(data_path='VisualNews/origin/', output_path='VisualNews/') 
extract_CLIP_features(data_path='Fakeddit/', output_path='Fakeddit/fd_original_') 
extract_CLIP_features(data_path='VisualNews/MISALIGN', output_path='VisualNews/MISALIGN_', use_image=False) 

# After extracting the Fakeddit we can create the Misalign dataset
prepare_Misalign(CLIP_VERSION="ViT-L/14", choose_gpu=0)

# Calculate the K most similar pairs from VisualNews. Necassary for creating CLIP-NESt. Can also be used to re-create CSt
get_K_most_similar(K_most_similar = 20, CLIP_VERSION="ViT-L/14", choose_gpu=1)

# Extract named entities from VisualNews pairs
extract_entities()

# Create the CLIP-NESt dataset and then extract its CLIP features
prepare_CLIP_NESt()
extract_CLIP_features(data_path='EntitySwaps_topic_clip', output_path='VisualNews/EntitySwaps_topic_clip', use_image=False) 

# Create the R-NESt dataset and then extract its CLIP features
prepare_R_NESt()
extract_CLIP_features(data_path='EntitySwaps_topic_random', output_path='VisualNews/EntitySwaps_topic_random', use_image=False)


# ### Table: I (Twitter)
run_experiment(
    dataset_methods_list = [
        'Twitter_comparable', # Uses the evaluation protocol of previous works
        'Twitter_corrected', # Uses a corrected evaluation protocol
    ],
    modality_options = [
        ["images", "texts"],
        ["texts"], 
        ["images"]
    ],
    epochs=30,
    seed_options = [0],
    lr_options = [5e-5, 1e-5],
    batch_size_options = [16],
    tf_layers_options = [1, 4],
    tf_head_options = [2, 8],
    tf_dim_options = [128, 1024],
    use_multiclass = False, 
    balancing_method = None,
    choose_gpu = 0, 
    init_model_name = ''
)

# ### Tables: II, III, IV (single datasets)
run_experiment(
    dataset_methods_list = [
        'random_sampling_topic', # RSt
        'clip_based_sampling_topic', # CSt
        'news_clippings_txt2txt', # NC-t2t
        'meir', 
        'EntitySwaps_random_topic', # R-NESt
        'EntitySwaps_CLIP_topic', # CLIP-NESt
        'fakeddit_original',
        'Misalign', 
        'Misalign_D', # 'downsample' is automatically applied
    ],
    modality_options = [
        ["images", "texts"],
        ["texts"], 
        ["images"]
    ],
    epochs=30,
    seed_options = [0],
    lr_options = [5e-5],
    batch_size_options = [512],
    tf_layers_options = [1, 4],
    tf_head_options = [2, 8],
    tf_dim_options = [128, 1024],
    use_multiclass = False, 
    balancing_method = None,
    choose_gpu = 0, 
    init_model_name = ''
)

# Table IV: Ensemble datasets for binary classification on FIGMENTS
run_experiment(
    dataset_methods_list = [
        'EntitySwaps_CLIP_topicXnews_clippings_txt2txt',     
        'EntitySwaps_random_topicXnews_clippings_txt2txt',     
        'MisalignXnews_clippings_txt2txt',  
        'Misalign_DXnews_clippings_txt2txt',
        'EntitySwaps_random_topicXMisalign_DXnews_clippings_txt2txt',
        'EntitySwaps_CLIP_topicXMisalign_DXnews_clippings_txt2txt',
        'EntitySwaps_random_topicXMisalignXnews_clippings_txt2txt',
        'EntitySwaps_CLIP_topicXMisalignXnews_clippings_txt2txt',        
    ],
    epochs=30,
    use_multiclass = False,
    balancing_method = 'downsample',
)

# Table V: Multiclass classification on FIGMENTS
run_experiment(
    dataset_methods_list = [
        'EntitySwaps_CLIP_topicXclip_based_sampling_topic',
        'EntitySwaps_CLIP_topicXnews_clippings_txt2txt',     
        'EntitySwaps_random_topicXclip_based_sampling_topic',
        'EntitySwaps_random_topicXnews_clippings_txt2txt',     
        'MisalignXclip_based_sampling_topic',
        'MisalignXnews_clippings_txt2txt',  
        'Misalign_DXclip_based_sampling_topic',
        'Misalign_DXnews_clippings_txt2txt',
        'EntitySwaps_random_topicXMisalign_DXnews_clippings_txt2txt',
        'EntitySwaps_CLIP_topicXMisalign_DXnews_clippings_txt2txt',
        'EntitySwaps_random_topicXMisalignXnews_clippings_txt2txt',
        'EntitySwaps_CLIP_topicXMisalignXnews_clippings_txt2txt',        
    ],
    epochs=30,
    use_multiclass = True,
    balancing_method = 'downsample',
)