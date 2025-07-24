import os
import re
import numpy as np
import pandas as pd
import cv2
import itertools
import json
from pyparsing import Combine
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sympy import ImageSet
import torch
import os
import sys
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from clip import load
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo
from detectron2.structures import Boxes
from clip_sim_filter import *
from evaluate import load
bert_scorer = load("bertscore")

def setup_detectron2_model(person_detector_threshold = 0.98):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = person_detector_threshold
    return DefaultPredictor(cfg)

predictor = setup_detectron2_model()

def detect_person(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        outputs = predictor(image_np)
        instances = outputs["instances"]
        pred_classes = instances.pred_classes

        if any(cls == 0 for cls in pred_classes):
            return True
        return False
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return False


def clean_repeated_parts(sentences):
    sentences = [s.strip().lower() for s in sentences]
    cleaned_sentences = []
    
    for i, current_sentence in enumerate(sentences):
        if i == 0:  
            cleaned_sentences.append(current_sentence)
        else:
            current_sub_sentences = [s.strip() for s in current_sentence.split('.') if s.strip()]
            cleaned_sub_sentences = []
            
            for sub_sentence in current_sub_sentences:
                is_repeated = False
                for prev_sentence in sentences[:i]:
                    prev_sub_sentences = [s.strip() for s in prev_sentence.split('.') if s.strip()]
                    if any(sub_sentence == prev_sub for prev_sub in prev_sub_sentences):
                        is_repeated = True
                        break
                
                if not is_repeated:
                    cleaned_sub_sentences.append(sub_sentence)
            
            cleaned_sentence = '. '.join(cleaned_sub_sentences)
            if cleaned_sentence:  
                cleaned_sentences.append(cleaned_sentence + '.')
    
    return cleaned_sentences

# Text-based filters
def is_single_sentence(text):
    text = text.strip()
    sentence_endings = re.compile(r'[.!?]')
    endings = sentence_endings.findall(text)
    num_endings = len(endings)
    if num_endings == 0:
        return True
    elif num_endings == 1:
        return text.endswith(('.', '!', '?'))
    else:
        return False


def no_stative_verbs(text):
    with open("/process_data/stative_verbs.json", 'r') as file:
        stative_verb_dict = json.load(file)
    
    stative_verbs = [item for sublist in stative_verb_dict.values() for item in sublist]
    
    text_lower = text.lower()
    
    pattern = r'\b(?:' + '|'.join(re.escape(verb) for verb in stative_verbs) + r')\b'
    
    if re.search(pattern, text_lower):
        return False
    else:
        return True

    
def not_start_with_pronoun(text):
    text = text.lower().strip()

    pattern = r'^(it|they|he|she)\b'

    return not bool(re.match(pattern, text))

def check_available_single_sentence(df):
    text_columns = [col for col in df.columns if col.startswith('text')]
    mask = df.apply(lambda row: all(is_single_sentence(str(row[text_col])) for text_col in text_columns), axis=1)
    
    available_df = df[mask].reset_index(drop=True)
    
    available_count = len(available_df)
    
    print(f"Number of rows where all text column is a single sentence: {available_count}/{len(df)}")
    
    return available_df

# def check_available_be_expression(df):
#     text_columns = [col for col in df.columns if col.startswith('text')]
#     mask = df.apply(lambda row: all(no_be_expression(str(row[text_col])) for text_col in text_columns), axis=1)
    
#     available_df = df[mask].reset_index(drop=True)
    
#     available_count = len(available_df)
    
#     print(f"Number of rows where no text column contains 'be' expression: {available_count}/{len(df)}")
    
#     return available_df

def check_available_non_stative_expression(df):
    text_columns = [col for col in df.columns if col.startswith('text')]
    mask = df.apply(lambda row: all(no_stative_verbs(str(row[text_col])) for text_col in text_columns), axis=1)
    available_df = df[mask].reset_index(drop=True)
    available_count = len(available_df)
    print(f"Number of rows where no text column contains stative verbs: {available_count}/{len(df)}")
    
    return available_df


def check_available_pronoun(df):
    text_columns = [col for col in df.columns if col.startswith('text')]
    mask = df.apply(lambda row: all(not_start_with_pronoun(str(row[text_col])) for text_col in text_columns), axis=1)
    
    available_df = df[mask].reset_index(drop=True)
    
    available_count = len(available_df)
    
    print(f"Number of rows where no text column starts with a pronoun: {available_count}/{len(df)}")
    
    return available_df


def check_available_no_similar_text(df, text_sim_threshold, text_thres_list, use_bertscore = True):
    
    text_columns = [col for col in df.columns if col.startswith('text')]
    no_similar_indices = []
    if use_bertscore:
        assert len(text_thres_list) == 6
        p_thres_1 = text_thres_list[0]
        r_thres_1 = text_thres_list[1]
        f_thres_1 = text_thres_list[2]
        p_thres_2 = text_thres_list[3]
        r_thres_2 = text_thres_list[4]
        f_thres_2 = text_thres_list[5]
        print(f'p_thres_1: {p_thres_1}, r_thres_1: {r_thres_1}, f_thres_1: {f_thres_1}')
        print(f'p_thres_2: {p_thres_2}, r_thres_2: {r_thres_2}, f_thres_2: {f_thres_2}')
    
    for index, row in df.iterrows():
        texts = row[text_columns].dropna().apply(
            lambda x: re.sub(r'[^\w\s]', '', x.lower())
        ).tolist()
        
        if len(texts) < 2:
            continue
        
        is_similar = False
        if not use_bertscore:
            
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            similarities = cosine_similarity(tfidf_matrix)
            
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    if similarities[i][j] > text_sim_threshold:
                        is_similar = True
                        break
                if is_similar:
                    break
        else:
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    bert_score = bert_scorer.compute(predictions=[row[f'text{i}']],references = [row[f'text{j}']], lang='en')
             
                    if bert_score['precision'][0] > p_thres_1 or bert_score['recall'][0] > r_thres_1 or bert_score['f1'][0] > f_thres_1:
                        is_similar = True
                        break
                    if bert_score['precision'][0] > p_thres_2 and bert_score['recall'][0] > r_thres_2 and bert_score['f1'][0] > f_thres_2:
                        is_similar = True
                        break
                if is_similar:
                    break
        
    
        
        if is_similar == False:
            no_similar_indices.append(index)
    
    no_similar_text_df = df.loc[no_similar_indices].reset_index(drop=True)
    
    print(f'Non-similar-text seq count: {len(no_similar_text_df)}/{df.shape[0]}')
    
    return no_similar_text_df

# Image-based filters
def compute_histogram(image):

    if image is None:
        print("Error: Image not loaded.")
        return None
    else:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

def check_hist_similarity(images, threshold, print_similarity = False):
    histograms = [compute_histogram(img) for img in images]
    hist_similarities = []
    for (i, hist1), (j, hist2) in itertools.combinations(enumerate(histograms), 2):
        try:
            hist1_numeric = np.array(hist1, dtype=np.float64)
            hist2_numeric = np.array(hist2, dtype=np.float64)

            if np.isnan(hist1_numeric).any() or np.isnan(hist2_numeric).any():
                print(f"Skipping similarity calculation for pair ({i}, {j}) due to NaN values.")
                return False  

            sim = cosine_similarity([hist1_numeric], [hist2_numeric])[0][0]
            hist_similarities.append(((i, j), sim))
            if sim > threshold:
                return False

        except ValueError:
            print(f"Skipping similarity calculation for pair ({i}, {j}) due to conversion error.")
            return False
        
    if print_similarity:
        for (img_pair, sim) in hist_similarities:
            print(f"Histogram similarity between image{img_pair[0]} and image{img_pair[1]}: {sim:.4f}")
    return True

def check_clip_similarity(img_paths, threshold, print_similarity = False):
    is_less_similar = True
    
    for i in range(len(img_paths)):
        for j in range(i + 1, len(img_paths)):
            sim = calculate_clip_sim_img_img(img_paths[i], img_paths[j])
            if print_similarity:
                print(f"CLIP similarity between image{i} and image{j}: {sim:.4f}")
            if sim > threshold:
                is_less_similar = False
                break
        if not is_less_similar:
            break
    return is_less_similar
    

def check_available_no_similar_image(df, sim_thres, use_clip):
    no_similar_indices = []
    text_columns = [col for col in df.columns if col.startswith('text')]
    link_columns = [col for col in df.columns if col.startswith('link')]
    
    for index, item in df.iterrows():
        jpg_paths = df.loc[index, link_columns].dropna().tolist()
        images = [cv2.imread(jpg_path) for jpg_path in jpg_paths]

        texts = df.loc[index, text_columns].dropna().tolist()

        assert len(images) == len(texts)
        n = len(images)
        
        if use_clip:
            is_less_similar = check_clip_similarity(jpg_paths, sim_thres, False)
        else:
            is_less_similar = check_hist_similarity(images, sim_thres, False)

        if is_less_similar == True:
            no_similar_indices.append(index)

    no_similar_df = df.loc[no_similar_indices].reset_index(drop=True)

    print(f'Non-similar-image seq count: {len(no_similar_df)}/{df.shape[0]}')

    return no_similar_df

def check_pair_similarity(df, sim_thres):
    text_columns = [col for col in df.columns if col.startswith('text')]
    link_columns = [col for col in df.columns if col.startswith('link')]
    df['available_pairs'] = [[] for _ in range(len(df))]
    
    valid_indices = []
    
    for index, item in df.iterrows():
        jpg_paths = df.loc[index, link_columns].dropna().tolist()
        texts = df.loc[index, text_columns].dropna().tolist()
        
        assert len(jpg_paths) == len(texts)
        
        available_pairs = []
        
        for i in range(len(jpg_paths)):
            for j in range(i + 1, len(jpg_paths)):
                sim_ii = calculate_clip_sim_img_text(jpg_paths[i], texts[i])
                sim_jj = calculate_clip_sim_img_text(jpg_paths[j], texts[j])
                sim_ij = calculate_clip_sim_img_text(jpg_paths[i], texts[j])
                sim_ji = calculate_clip_sim_img_text(jpg_paths[j], texts[i])
                
                if sim_ii - sim_ij > sim_thres and sim_ii - sim_ji > sim_thres and sim_jj - sim_ij > sim_thres and sim_jj - sim_ji > sim_thres:
                    available_pairs.append((i, j))
        
        df.at[index, 'available_pairs'] = available_pairs
        if len(available_pairs) > 0:
            valid_indices.append(index)
            
    no_ambiguous_img_text = df.loc[valid_indices].reset_index(drop=True)
    
    print(f'No ambiguous img-text pair seq count: {len(no_ambiguous_img_text)}/{df.shape[0]}')
    
    return no_ambiguous_img_text 


def check_triple_similarity(df, sim_thres=0.0):
    text_columns = [col for col in df.columns if col.startswith('text')]
    link_columns = [col for col in df.columns if col.startswith('link')]
    df['available_triples'] = [[] for _ in range(len(df))]
    
    valid_indices = []
    available_count = 0
    for index, item in df.iterrows():
        jpg_paths = df.loc[index, link_columns].dropna().tolist()
        texts = df.loc[index, text_columns].dropna().tolist()
        if len(jpg_paths) == 0 or len(texts) == 0:
            continue
        
        # jpg_paths = [path.replace('/process_data', '') for path in jpg_paths]
        # if 'pororo' in jpg_paths[0] or 'flintstones' in jpg_paths[0]:
        #     jpg_paths = [path.replace('/scratch/song0018/temporal_data/', '/home/ysong/temporal-mllms/data/temporal_data/') for path in jpg_paths]
        
        try:
            for path in jpg_paths:
                with Image.open(path) as img:
                    img.verify()  # Verify it's a valid image
            # Try to actually load images to ensure they're not corrupted
            for path in jpg_paths:
                with Image.open(path) as img:
                    img.load()
        except (IOError, SyntaxError, OSError) as e:
            print(f"Skipping row {index}, error with image: {e}")
            continue
        
        assert len(jpg_paths) == len(texts)
        
        available_triples = []
        for x in range(len(jpg_paths)):
            for y in range(x + 1, len(jpg_paths)):
                for z in range(y + 1, len(jpg_paths)):
                    sim_xx = calculate_clip_sim_img_text(jpg_paths[x], texts[x])
                    sim_yy = calculate_clip_sim_img_text(jpg_paths[y], texts[y])
                    sim_zz = calculate_clip_sim_img_text(jpg_paths[z], texts[z])
                    sim_xy = calculate_clip_sim_img_text(jpg_paths[x], texts[y])
                    sim_xz = calculate_clip_sim_img_text(jpg_paths[x], texts[z])
                    sim_yz = calculate_clip_sim_img_text(jpg_paths[y], texts[z])
                    sim_yx = calculate_clip_sim_img_text(jpg_paths[y], texts[x])
                    sim_zx = calculate_clip_sim_img_text(jpg_paths[z], texts[x])
                    sim_zy = calculate_clip_sim_img_text(jpg_paths[z], texts[y])
                    # print(x,y,z)
                    # print(sim_xx, sim_yy, sim_zz, sim_xy, sim_xz, sim_yz, sim_yx, sim_zx, sim_zy)
                    
                    if sim_xx - sim_xy > sim_thres and sim_xx - sim_xz > sim_thres and sim_xx - sim_yx > sim_thres and sim_xx - sim_zx > sim_thres:
                        if sim_yy - sim_yx > sim_thres and sim_yy - sim_yz > sim_thres and sim_yy - sim_xy > sim_thres and sim_yy - sim_zy > sim_thres:
                            if sim_zz - sim_zx > sim_thres and sim_zz - sim_zy > sim_thres and sim_zz - sim_xz > sim_thres and sim_zz - sim_yz > sim_thres:
                                available_triples.append((x, y, z))
                                available_count += 1
                    
                    
        
        df.at[index, 'available_triples'] = available_triples

    
    print(f'Rows with available_triples: {available_count}')
    
    return df




def row_has_person(row, min_person_detections=2):
    #print("minimum num of images have person:", min_person_detections)
    person_count = 0
    for col in row.index:
        if col.startswith("link") and detect_person(row[col]):
            person_count += 1
            if person_count >= min_person_detections:
                return True
    return False

def check_at_least_one_person(df):
    filtered_df = df[df.apply(row_has_person, axis=1)]
    print(f'At least one person in image seq count: {len(filtered_df)}/{len(df)}')
    return filtered_df


def count_tokens(text):
    return len(text.split())

def select_shortest_row(group):
    shortest_rows_group = group[group['token_count'] == group['token_count'].min()]
    return shortest_rows_group.sample(n=1, random_state=1) 
    
def select_longest_row(group):
    shortest_rows_group = group[group['token_count'] == group['token_count'].max()]
    return shortest_rows_group.sample(n=1, random_state=1) 

def select_each_seq(df, criteria='imdb_id', shortest = True):
    
    df['token_count'] = df['full_text'].apply(count_tokens)
    
    if shortest:
        selected_rows = df.groupby(criteria).apply(select_shortest_row).reset_index(drop=True)
    else:
        selected_rows = df.groupby(criteria).apply(select_longest_row).reset_index(drop=True)

    assert df[criteria].nunique() == selected_rows.shape[0]
    print(selected_rows.shape[0])
    return selected_rows


# Apply data filters
def filter_data(df, img_sim_threshold, img_seq_only_criteria, select_shortest, clip_sim_threshold, use_bertscore, use_clipsim, text_sim_threshold, text_thres_list, check_person = False):    
    no_stative_verbs = check_available_non_stative_expression(df)
    pronoun = check_available_pronoun(no_stative_verbs)
    no_similar_text = check_available_no_similar_text(pronoun, text_sim_threshold, text_thres_list, use_bertscore)
    if check_person:
        at_least_one_person = check_at_least_one_person(no_similar_text)
        no_similar_img = check_available_no_similar_image(at_least_one_person, img_sim_threshold, use_clipsim)
    else:
        no_similar_img = check_available_no_similar_image(no_similar_text, img_sim_threshold, use_clipsim)
    no_ambiguous_img_text = check_pair_similarity(no_similar_img, clip_sim_threshold)
    final_df = select_each_seq(no_ambiguous_img_text, criteria=img_seq_only_criteria, shortest=select_shortest)
    
    return final_df


def get_combined_img_path(data_source, base_path, img_paths, use_number = True, reversed = False):
    if use_number:
        if reversed:
            combined_folder = "combined_reversed"
        else:
            combined_folder = "combined"
    else:
        combined_folder = "combined_shuffled"
        
    if data_source == 'vwp':
        img_paths = [path.strip() for path in img_paths]
        subset_info = img_paths[0].split('/')[-3]
        identifier = img_paths[0].split('/')[-2]
        shot_numbers = [path.split('/')[-1].split('_')[1] for path in img_paths]
        combined_shots = '_'.join(shot_numbers)
        combined_path = f"{base_path}/{combined_folder}/{subset_info}-{identifier}-shot_{combined_shots}.jpg"

    elif data_source == 'pororo':
        image_numbers = [path.split('/')[-1].split('.')[0] for path in img_paths]
        combined_shots = '_'.join(image_numbers)
        combined_path = f"{base_path}/{combined_folder}/images_{combined_shots}.png"

    elif data_source == 'flintstones':
        episode_info = img_paths[0].split('/')[-1].split('_shot_')[0]
        shot_numbers = [path.split('_shot_')[1].split('_')[0] for path in img_paths]
        combined_shots = '_'.join(shot_numbers)
        combined_path = f"{base_path}/{combined_folder}/{episode_info}_shot_{combined_shots}.jpg"
    
    elif data_source == 'vist':
        image_numbers = [path.split('/')[-1].split('.')[0] for path in img_paths]
        combined_images = '_'.join(image_numbers)
        combined_path = f"{base_path}/{combined_folder}/images_{combined_images}.jpg"
    else:
        print("Invalid data source")
        return None
        
        
    return combined_path
    

def combine_images(df, data_source, base_path):
    
    
    for index, row in df.iterrows():
    
        valid_pairs = []
        img_paths = []
        i = 0
        while f'text{i}' in row.index and f'link{i}' in row.index:
            if pd.notna(row[f'text{i}']) and pd.notna(row[f'link{i}']):
                valid_pairs.append(i)
                img_paths.append(row[f'link{i}'])
            i += 1
        
        if not valid_pairs:
            print("No valid text-image pairs found")
            continue
        
        img_paths = [row[f'link{i}'] for i in valid_pairs]

        if data_source == 'vwp':
            font_size = 24
            y_padding = 30
            combined_path = get_combined_img_path(data_source, base_path, img_paths)
        elif data_source == 'pororo':
            font_size = 10
            y_padding = 15
            combined_path = get_combined_img_path(data_source, base_path, img_paths)
        elif data_source == 'flintstones':
            font_size = 10
            y_padding = 15
            combined_path = get_combined_img_path(data_source, base_path, img_paths)
        elif data_source == 'vist':
            font_size = 60
            y_padding = 15
            combined_path = get_combined_img_path(data_source, base_path, img_paths)
        else:
            print("Invalid data source")
            continue
            
        padding = 10  
        
        if data_source == 'vist':
            images = [Image.open(path).resize((443, 240), Image.Resampling.LANCZOS) for path in img_paths]
        else:
            images = [Image.open(path) for path in img_paths]
        total_width = sum(img.width for img in images) + padding * (len(images) + 1)  
        max_height = max(img.height for img in images) + (padding * 2) 
        
        combined_image = Image.new('RGB', (total_width, max_height + 30), 'white')  
        
        x_offset = padding  
        for idx, img in enumerate(images):
            y_offset = padding + y_padding  
            combined_image.paste(img, (x_offset, y_offset))
            
            draw = ImageDraw.Draw(combined_image)
            try:
                font = ImageSet.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except:
                font = ImageFont.load_default()
                
            text = 'image '+ str(idx+1)
            text_width = draw.textlength(text, font=font)
            text_x = x_offset + (img.width - text_width) // 2
            draw.text((text_x, 5), text, fill="black", font=font)
            
            x_offset += img.width + padding
        
        os.makedirs(os.path.dirname(combined_path), exist_ok=True) 
        combined_image.save(combined_path)

