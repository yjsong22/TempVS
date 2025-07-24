import os
from numpy import character
import pandas as pd
from extract_events import is_name_or_proper_noun
from data_filters import get_combined_img_path
import math
from itertools import product
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageFile
from sympy import ImageSet
import sys
import string
import random
import itertools

def shuffle_list(original_list, seed_value = 56):
    
    random.seed(seed_value)

    indexed_list = list(enumerate(original_list, start=1))

    indices = list(range(len(original_list)))  
    shuffled_indices = indices[:]             
    random.shuffle(shuffled_indices)          

    shuffled_list_for_question = [original_list[i] for i in shuffled_indices]

    answer_order = [shuffled_indices.index(i) + 1 for i in range(len(original_list))]
    
    answer_order = [chr(96 +  num) for num in answer_order]

    return shuffled_list_for_question, answer_order

def create_candidate_options(original_list, prefix = "Image", gt_answer_list=None):
    assert len(original_list) == len(gt_answer_list)
    all_permutations = list(itertools.permutations(original_list))

    random.shuffle(all_permutations)

    selected_orders = all_permutations[:10]
    
    orders = [list(order) for order in selected_orders]
    
    left_orders = []
    for order in orders:
        if order != gt_answer_list:
            left_orders.append(order)
        if len(left_orders) == 4:
            break
    
    assert gt_answer_list not in left_orders        
    options = [" -> ".join([f"{prefix} {option}" for option in order]) for order in left_orders]
    assert len(options) == 4
    
    return options

def generate_option_string(options, gt_answer, prefix, global_index):
    gt_string = " -> ".join([f"{prefix} {option}" for option in gt_answer])
    all_options = options.copy()
    
    insert_pos = global_index % 5
    all_options.insert(insert_pos, gt_string)
    
    option_string = ""
    for i, option in enumerate(all_options):
        letter = chr(65 + i)  
        option_string += f"{letter}. {option}"
        if i < len(all_options) - 1:
            option_string += ";\n"
        
    
    return option_string, chr(65+insert_pos)
    


def combine_image_seq(data_source, base_path, img_paths, use_number= True, reversed=False):

    if data_source == 'vwp':
        font_size = 24
        y_padding = 30
        combined_path = get_combined_img_path(data_source, base_path, img_paths, use_number=use_number, reversed=reversed)
    elif data_source == 'pororo':
        font_size = 10
        y_padding = 15
        combined_path = get_combined_img_path(data_source, base_path, img_paths, use_number=use_number, reversed=reversed)
    elif data_source == 'flintstones':
        font_size = 10
        y_padding = 15
        combined_path = get_combined_img_path(data_source, base_path, img_paths, use_number=use_number, reversed=reversed)
    elif data_source == 'vist':
        font_size = 60
        y_padding = 15
        combined_path = get_combined_img_path(data_source, base_path, img_paths, use_number=use_number, reversed=reversed)
    else:
        print("Invalid data source")
        sys.exit(0)
    
    if use_number == False:
        assert 'shuffled' in combined_path
    if reversed:
        assert 'reversed' in combined_path
    
        
    padding = 10  
    
    if data_source == 'vist':
        images = []
        for path in img_paths:
            try:
                img = Image.open(path).resize((443, 240), Image.Resampling.LANCZOS)
                images.append(img)
            except FileNotFoundError:
                print(f"Warning: Image not found at {path}, skipping...")
    else:
        images = []
        for path in img_paths:
            try:
                img = Image.open(path)
                images.append(img)
            except FileNotFoundError:
                print(f"Warning: Image not found at {path}, skipping...")
    
    # Only proceed if we have at least one valid image
    if not images:
        print("No valid images found, skipping this combination")
        return None
    
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
        
        if use_number:
            if reversed:
                text = 'Image '+ str(len(images) - idx)
            else:
                text = 'Image '+ str(idx + 1)
        else:
            text = 'Image ' + str(chr(96 + idx + 1))
        text_width = draw.textlength(text, font=font)
        text_x = x_offset + (img.width - text_width) // 2
        draw.text((text_x, 5), text, fill="black", font=font)
        
        x_offset += img.width + padding
    
    os.makedirs(os.path.dirname(combined_path), exist_ok=True) 
    combined_image.save(combined_path)
    
    
def combine_image_pair_triple(data_source, base_path, img_paths):
    assert len(img_paths) == 2 or len(img_paths) == 3

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
        sys.exit(0)
    
        
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
            
            
        ordinal_map = {1: 'First', 2: 'Second', 3: 'Third'}
        text = f'{ordinal_map[idx + 1]} image'
        
        text_width = draw.textlength(text, font=font)
        text_x = x_offset + (img.width - text_width) // 2
        draw.text((text_x, 5), text, fill="black", font=font)
        
        x_offset += img.width + padding
    
    os.makedirs(os.path.dirname(combined_path), exist_ok=True) 
    combined_image.save(combined_path)


def load_templates():
    templates = {}
    for filename in ['statement_types_paired.json',  'statement_types_triple.json', 
                     'option_templates.json', 'task_instructions.json', 
                     'requirement_templates.json', 'answer_format_templates.json']:
        with open("prompt_templates/"+filename, 'r') as f:
            templates[filename.replace('.json', '')] = json.load(f)
    return templates

templates = load_templates()
statement_types_paired = templates['statement_types_paired']
statement_types_triple = templates['statement_types_triple']
option_templates = templates['option_templates']
task_instructions = templates['task_instructions']
requirement_templates = templates['requirement_templates'] 
answer_format_templates = templates['answer_format_templates']

def get_gt_answer_statement(statement_group,task_name, option_index):
    options = {}
    option_template = option_templates[task_name][option_index].strip()[:-1]
    for option in option_template.split(';'):
        key, value = option.strip().split('. ')
        options[value.lower()] = key.strip()

    if statement_group.endswith('_gt'):
        return options['true']
    elif statement_group.endswith('_nc'):
        return options['false']
    else:
        return None
    
def get_gt_answer_paired_triple_grounding(gt_index,task_name, option_index):
    options = {}
    option_template = option_templates[task_name][option_index].strip()[:-1]
    for option in option_template.split(';'):
        key, value = option.strip().split('. ')
        options[value.lower()] = key.strip()

    if task_name == 'paired_grounding_one_text':
        if gt_index == 0:
            return options['first image']
        elif gt_index == 1:
            return options['second image']
        else:
            return None
    elif task_name == 'paired_grounding_one_image':
        if gt_index == 0:
            return options['first event']
        elif gt_index == 1:
            return options['second event']
        else:
            return None
    elif task_name == 'triple_grounding_one_text':
        if gt_index == 0:
            return options['first image']
        elif gt_index == 1:
            return options['second image']
        elif gt_index == 2:
            return options['third image']
        else:
            return None
    elif task_name == 'triple_grounding_one_image':
        if gt_index == 0:
            return options['first event']
        elif gt_index == 1:
            return options['second event']
        elif gt_index == 2:
            return options['third event']
        else:
            return None
    else:
        return None
    
def create_statements_pair(event_pre, event_post, statement_group):
    if statement_group == 'after_gt':
        return f"{event_post.capitalize().rstrip('.,;!?')} after {event_pre[0].lower() + event_pre[1:].rstrip('.,;!?') + '.' if not is_name_or_proper_noun(event_pre) else event_pre.rstrip('.,;!?') + '.'}"
    elif statement_group == 'after_nc':
        return f"{event_pre.capitalize().rstrip('.,;!?')} after {event_post[0].lower() + event_post[1:].rstrip('.,;!?') + '.' if not is_name_or_proper_noun(event_post) else event_post.rstrip('.,;!?') + '.'}"
    elif statement_group == 'before_gt':
        return f"{event_pre.capitalize().rstrip('.,;!?')} before {event_post[0].lower() + event_post[1:].rstrip('.,;!?') + '.' if not is_name_or_proper_noun(event_post) else event_post.rstrip('.,;!?') + '.'}"
    elif statement_group == 'before_nc':
        return f"{event_post.capitalize().rstrip('.,;!?')} before {event_pre[0].lower() + event_pre[1:].rstrip('.,;!?') + '.' if not is_name_or_proper_noun(event_pre) else event_pre.rstrip('.,;!?') + '.'}"
    elif statement_group == 'then_gt':
        return f"{event_pre.capitalize().rstrip('.,;!?') + '.'} Then, {event_post[0].lower() + event_post[1:].rstrip('.,;!?') + '.' if not is_name_or_proper_noun(event_post) else event_post.rstrip('.,;!?') + '.'}"
    elif statement_group == 'then_nc':
        return f"{event_post.capitalize().rstrip('.,;!?') + '.'} Then, {event_pre[0].lower() + event_pre[1:].rstrip('.,;!?') + '.' if not is_name_or_proper_noun(event_pre) else event_pre.rstrip('.,;!?') + '.'}"
    elif statement_group == 'earlier_gt':            
        return f"{event_post.capitalize().rstrip('.,;!?') + '.'} Earlier, {event_pre[0].lower() + event_pre[1:].rstrip('.,;!?') + '.' if not is_name_or_proper_noun(event_pre) else event_pre.rstrip('.,;!?') + '.'}"
    elif statement_group == 'earlier_nc':
        return f"{event_pre.capitalize().rstrip('.,;!?') + '.'} Earlier, {event_post[0].lower() + event_post[1:].rstrip('.,;!?') + '.' if not is_name_or_proper_noun(event_post) else event_post.rstrip('.,;!?') + '.'}"
    elif statement_group ==  'implictpair_gt':
        return f"{event_pre.capitalize().rstrip('.,;!?') + '.'} {event_post.capitalize().rstrip('.,;!?') + '.'}"
    elif statement_group ==  'implictpair_nc':
        return f"{event_post.capitalize().rstrip('.,;!?') + '.'} {event_pre.capitalize().rstrip('.,;!?') + '.'}"
    else:
        return None
    
def create_statements_triple(event_pre, event_mid, event_post, statement_group, seed_value = 56):
    shuffled_lists = [[event_mid, event_pre, event_post],
                      [event_mid, event_post, event_pre],
                      [event_pre, event_post, event_mid],
                      [event_post, event_pre, event_mid],
                      [event_post, event_mid, event_pre]]
    if statement_group == 'beforeafter_gt':
        events_seq = [event_pre, event_mid, event_post]
        return f"{events_seq[0].capitalize().rstrip('.,;!?')} before {events_seq[1][0].lower() + events_seq[1][1:].rstrip('.,;!?') if not is_name_or_proper_noun(events_seq[1]) else events_seq[1].rstrip('.,;!?') + ','} and after that, {events_seq[2][0].lower() + events_seq[2][1:].rstrip('.,;!?') + '.' if not is_name_or_proper_noun(events_seq[2]) else events_seq[2].rstrip('.,;!?') + '.'}"
    elif statement_group == 'beforeafter_nc':
        random.seed(seed_value)
        events_seq = random.choice(shuffled_lists)
        return f"{events_seq[0].capitalize().rstrip('.,;!?')} before {events_seq[1][0].lower() + events_seq[1][1:].rstrip('.,;!?') if not is_name_or_proper_noun(events_seq[1]) else events_seq[1].rstrip('.,;!?') + ','} and after that, {events_seq[2][0].lower() + events_seq[2][1:].rstrip('.,;!?') + '.' if not is_name_or_proper_noun(events_seq[2]) else events_seq[2].rstrip('.,;!?') + '.'}"
        
    elif statement_group == 'laterfinally_gt':
        events_seq = [event_pre, event_mid, event_post]
        return f"{events_seq[0].capitalize().rstrip('.,;!?')+ '.'} Later, {events_seq[1][0].lower() + events_seq[1][1:].rstrip('.,;!?') if not is_name_or_proper_noun(events_seq[1]) else events_seq[1].rstrip('.,;!?')}. Thereafter, {events_seq[2][0].lower() + events_seq[2][1:].rstrip('.,;!?') + '.' if not is_name_or_proper_noun(events_seq[2]) else events_seq[2].rstrip('.,;!?') + '.'}"
    elif statement_group == 'laterfinally_nc':
        random.seed(seed_value)
        events_seq = random.choice(shuffled_lists)
        return f"{events_seq[0].capitalize().rstrip('.,;!?')+ '.'} Later, {events_seq[1][0].lower() + events_seq[1][1:].rstrip('.,;!?') if not is_name_or_proper_noun(events_seq[1]) else events_seq[1].rstrip('.,;!?')}. Thereafter, {events_seq[2][0].lower() + events_seq[2][1:].rstrip('.,;!?') + '.' if not is_name_or_proper_noun(events_seq[2]) else events_seq[2].rstrip('.,;!?') + '.'}"
    
    elif statement_group == 'firstsecondthird_gt':
        events_seq = [event_pre, event_mid, event_post]
        return f"First, {events_seq[0].capitalize().rstrip('.,;!?')+ '.'} Second, {events_seq[1][0].lower() + events_seq[1][1:].rstrip('.,;!?') if not is_name_or_proper_noun(events_seq[1]) else events_seq[1].rstrip('.,;!?')}. Third, {events_seq[2][0].lower() + events_seq[2][1:].rstrip('.,;!?') + '.' if not is_name_or_proper_noun(events_seq[2]) else events_seq[2].rstrip('.,;!?') + '.'}"
    elif statement_group == 'firstsecondthird_nc':
        random.seed(seed_value)
        events_seq = random.choice(shuffled_lists)
        return f"First, {events_seq[0].capitalize().rstrip('.,;!?')+ '.'} Second, {events_seq[1][0].lower() + events_seq[1][1:].rstrip('.,;!?') if not is_name_or_proper_noun(events_seq[1]) else events_seq[1].rstrip('.,;!?')}. Third, {events_seq[2][0].lower() + events_seq[2][1:].rstrip('.,;!?') + '.' if not is_name_or_proper_noun(events_seq[2]) else events_seq[2].rstrip('.,;!?') + '.'}"
    
    elif statement_group == 'implicttriple_gt':
        events_seq = [event_pre, event_mid, event_post]
        return f"{events_seq[0].capitalize().rstrip('.,;!?')+ '.'} {events_seq[1].rstrip('.,;!?') if not is_name_or_proper_noun(events_seq[1]) else events_seq[1].rstrip('.,;!?')}. {events_seq[2].rstrip('.,;!?') + '.' if not is_name_or_proper_noun(events_seq[2]) else events_seq[2].rstrip('.,;!?') + '.'}"
    elif statement_group == 'implicttriple_nc':  
        random.seed(seed_value)
        events_seq = random.choice(shuffled_lists)  
        return f"{events_seq[0].capitalize().rstrip('.,;!?')+ '.'} {events_seq[1].rstrip('.,;!?') if not is_name_or_proper_noun(events_seq[1]) else events_seq[1].rstrip('.,;!?')}. {events_seq[2].rstrip('.,;!?') + '.' if not is_name_or_proper_noun(events_seq[2]) else events_seq[2].rstrip('.,;!?') + '.'}"
    
    else:
        return None
    
def validate_image_paths(img_path_list):
    """Check if all image paths exist and are valid image files."""
    for img_path in img_path_list:
        if not os.path.exists(img_path):
            return False
        try:
            # Try to open the image to verify it's a valid image file
            with Image.open(img_path) as img:
                img.verify()  # Verify it's a valid image
        except Exception:
            return False
    return True

    
def generate_combinations(list1, list2, list3):
    return [' '.join(combination) for combination in product(list1, list2, list3)]

def find_characters_in_string(input_string, subset):
    with open(f"characters_{subset}.json", "r") as file:
        character_dict = json.load(file) 
    input_string_lower = input_string.lower()
    matches = [description for key, description in character_dict.items() if key in input_string_lower]
    return matches

def create_questions(df, data_source, task_name='paired_event_discrimination'):
    blind_instruction_statement = ["Analyze the given statement with respect to temporal logic and return either 'True' or 'False'. No need to give reasoning process. Submit only the right option letter as your answer, e.g., Option [Letter]. ",
                                   "Using temporal reasoning, determine if the statement below is True or False. Respond only with the right option letter from the provided options. Do not tell the reasons of your decision. ",
                                   "Evaluate the truth value of the following statement based on temporal logic. Provide the most suitable choice letter in the format of 'Option [Letter]' as your response only without additional explaination."
                                   ]
    #blind_instruction_statement_reason = "Focus on the statement text only and ignore the image. Determine whether the statement is True, False, or Unsure in terms of temporal logic. \nFirst, thinc about the reasoning process step by step. Enclose the reasoning process within <think> and </think> tags. \nThen, provide the answer (True, False, or Unsure) within <answer> and </answer> tags.\nDo not include any additional information outside these tags."
    blind_instruction_ordersent_opg = ["You are tasked with organizing a set of shuffled sentences into a coherent and temporally logical order. Your goal is to determine the correct sequence of the sentences based on logical flow and temporal progression. \nOnce you have determined the correct order, provide the answer directly in the format: 'Sentence y -> Sentence x -> Sentence v -> Sentence w -> Sentence z'. \nEnclose your answer within <answer> and </answer> tags.",
                                       "Reorder the given shuffled sentences into a coherent and temporally logical sequence. Focus only on the provided sentences and their logical and temporal relationships. Determine the correct order based on logical flow and temporal progression.\nOnce you have determined the correct sequence, provide your answer directly in the format: 'Sentence h -> Sentence j -> Sentence i -> Sentence l -> Sentence k'. \nEnclose your answer within <answer> and </answer> tags.",
                                       "Arrange the following shuffled sentences into a logically and temporally coherent sequence. Consider only the given sentences and their internal relationships to determine the correct order of the events based on logical flow and temporal progression. \nOnce you have determined the correct sequence, output your answer in the format: 'Sentence p -> Sentence o -> Sentence r -> Sentence q -> Sentence s'. \nEnsure your response is enclosed within <answer> and </answer> tags."
                                       ]
    
    #blind_instruction_ordersent_opg_reason = "You are tasked with organizing a set of shuffled sentences into a coherent and temporally logical order. Focus solely on the provided sentences and ignore the image. Your goal is to determine the correct sequence of the sentences based on logical flow and temporal progression. \nBefore providing the final answer, think through the reasoning process step by step. Enclose your reasoning process within <think> and </think> tags. Once you have determined the correct order, provide the answer in the format 'Sentence y -> Sentence x -> Sentence v -> Sentence w -> Sentence z', enclosed within <answer> and </answer> tags."
    blind_instruction_ordersent_opt = ["Rearrange the shuffled sentences into a coherent and temporally logical sequence. Select the correct sequence from the multiple-choice options and respond with 'Option [letter]'. Do not include any explanations.",
                                       "Sort the scrambled sentences into a coherent, meaningful and temporally logical order. Pick the correct sequence from the multiple-choice answers and respond only with 'Option [letter]'.",
                                       "Arrange the shuffled sentences into a coherent, fluent, well-structured and temporally logical sequence. Choose the correct answer from the multiple-choice options and format your response as 'Option [letter]', with no extra details."
                                       ]
    #blind_instruction_ordersent_opt_reason = "Organize the shuffled sentences into a coherent, temporally logical sequence. Focus only on the provided texts, ignoring external context. Determine the correct order based on logical flow and temporal progression. \nFirst, think through the reasoning process step by step, analyzing how the sentences connect. Then, provide the final answer by selecting the correct sequence from the multiple-choice options. \nYour response must include: 1. A detailed reasoning process enclosed in <think> and </think> tags. 2. The final answer, enclosed in <answer> and </answer> tags."
    
    statement_instruction_cot = ["Analyze the provided image sequence to determine whether the following statement is True or False. When making the choice, carefully examine the evidence presented in the sequence of images from left to right. First, describe the key details and changes observed in each image. Then, explain how these details support or contradict the given statement. Finally, based on your step-by-step reasoning, conclude whether the statement is True or False. Ensure your response follows this format: the reasoning process should be enclosed within <think> and </think> tags, and the final answer should be enclosed within <answer> and </answer> tags.",
                                 "Examine the sequence of images provided and evaluate whether the given statement is True or False. Pay close attention to the details in each image and how they evolve from left to right. Use a logical, step-by-step approach to analyze the evidence and justify your conclusion. \nYour response must include: 1. A detailed reasoning process enclosed within <think> and </think> tags. 2. A final answer enclosed within <answer> and </answer> tags."]
    ordersent_instruction_opg_cot = ["Presented is an image sequence in the order from left to right along with several unordered sentences. Your task is to determine the correct sequence of the sentences by analyzing the context, events, or details depicted in the images. Follow this process step by step to arrive at your answer: \n1.Carefully examine each image in the sequence from left to right, noting key visual elements, actions, or changes.\n2. Analyze the content of each sentence and identify how it relates to the details in the images.\n3. Use the visual evidence to deduce the logical or chronological order of the sentences, ensuring your reasoning is grounded in the image sequence.\n4. Avoid making assumptions that are not supported by the images or sentences.\n5. Format your response as follows: Enclose your step-by-step reasoning process within <think> and </think> tags. Enclose the final ordered sequence of sentences (e.g., Sentence g -> Sentence f -> Sentence i -> Sentence h -> Sentence j) within <answer> and </answer> tags.",
                                     "You are given an ordered sequence of images and several sentences presented in random order. Your task is to reason step-by-step to rearrange the sentences into their correct chronological and logical order based on the content of the images. \nFollow these instructions carefully:\n1. Analyze the content of the image sequence from left to right.\n2. Use the visual information to guide your understanding and match the sentences to their logical or chronological positions.\n3. Avoid making assumptions not directly supported by the images or sentences.\nFor clarity, format your response as follows: Enclose your step-by-step reasoning process within <think> and </think> tags.Enclose the final ordered sequence of sentences (e.g., Sentence j -> Sentence h -> Sentence i -> Sentence f -> Sentence g) within <answer> and </answer> tags.\nFocus on reasoning explicitly for each decision to ensure a clear and logical explanation.",
                                     "You are presented with a left-to-right image sequence and several sentences in random order. Your task is to determine the correct order of the sentences based on the context, events, or details observable in the images. Think step by step, using the content of the images to guide your reasoning. Avoid assumptions not supported by the image sequence or the sentences.\nFormat your response as follows: Enclose your step-by-step reasoning process within <think> and </think> tags. Enclose the final ordered sequence of sentences (e.g., Sentence j -> Sentence h -> Sentence i -> Sentence f -> Sentence g) within <answer> and </answer> tags."]
    ordersent_instruction_opt_cot = ["You are given an ordered sequence of images and several sentences presented in random order. Your task is to reason step-by-step to select the correct chronological and logical order of the sentences based on the content of the images. Follow these instructions carefully:\n1. Analyze the content of the image sequence from left to right.\n2. Use the visual information to guide your understanding and match the sentences to their correct order.\n3. Avoid making assumptions not directly supported by the images or the sentences.\n4. For clarity, format your response as follows: Enclose your step-by-step reasoning process within <think> and </think> tags. Enclose your answer of the correct option (e.g., Option A, Option B) within <answer> and </answer> tags.\nFocus on explicit reasoning for each decision to ensure a clear and logical explanation.",
                                     "Analyze the provided image sequence to determine the correct chronological and logical order of the sentences. Your task is to reason step-by-step to select the correct sequence from the multiple-choice options based on the content of the images. Follow these instructions carefully:\n1. Analyze the content of the image sequence from left to right.\n2. Use the visual information to guide your understanding and match the sentences to their correct order.\n3. Avoid making assumptions not directly supported by the images or the sentences.\n4. For clarity, format your response as follows: Enclose your step-by-step reasoning process within <think> and </think> tags. Enclose your answer of the correct option (e.g., Option A, Option B) within <answer> and </answer> tags.\nFocus on explicit reasoning for each decision to ensure a clear and logical explanation.",
                                     "You are presented with an ordered image sequence and several sentences in random order. Your task is to determine the correct order of the sentences based on the context, events, or details observable in the images. Multiple-choice options are provided, with each option representing a possible sequence of the sentences. Think step by step, using the content of the images to guide your reasoning. Avoid assumptions not supported by the image sequence or the sentences.\nFormat your response as follows: Enclose your step-by-step reasoning process within <think> and </think> tags. Enclose the selected option (e.g., Option A, Option B) within <answer> and </answer> tags."]
    orderimg_instruction_opg_cot = ["You are presented with a set of shuffled images and a pragraph of text. Your task is to determine the correct order of the images by analyzing the content of the sequential sentences. Follow this process step by step to arrive at your answer: \n1. Carefully examine each image and note key visual elements, actions, or changes.\n2. Analyze the content of each sentence and identify how it relates to the details in the images.\n3. Use the textual evidence to deduce the logical or chronological order of the images, ensuring your reasoning is grounded in the sentence content.\n4. Avoid making assumptions that are not supported by the images or sentences.\n5. Format your response as follows: Enclose your step-by-step reasoning process within <think> and </think> tags. Enclose the final ordered sequence of images (e.g., Image y -> Image x -> Image v -> Image w -> Image z) within <answer> and </answer> tags.",
                                    "Let’s think step by step. You are given a set of images and a story. Your task is to rearrange the images in the correct order based on the content of the story. First, carefully read the provided text, paying close attention to the events, actions, and details described. Next, analyze each image and identify how it corresponds to specific parts of the story. Then, logically reorder the images to match the sequence of events in the story. Take your time to ensure the images align with the narrative flow, focusing on key moments and transitions. Finally, verify your arrangement by cross-checking the images with the story to confirm accuracy. Enclose your reasoning process within <think> and </think> tags. Enclose your final ordered sequence of images (e.g., Image y -> Image x -> Image v -> Image w -> Image z) within <answer> and </answer> tags.",
                                    "Reorder the images into the correct sequence based on events described in the text. Use multimodal reasoning by identifying key events and details from the text to guide your decisions. As you solve the task, explain your step-by-step reasoning process clearly and systematically, enclosing it within <think> and </think> tags. Provide the final ordered sequence of images (e.g., Image y -> Image x -> Image v -> Image w -> Image z) within <answer> and </answer> tags."]
    orderimg_instruction_opt_cot = ["Rearrange the following images in the correct sequence based on the content of the story. Carefully review the provided text. Pay close attention to the events, actions, and details described to logically reorder the images. Use <think> tags to explain your reasoning process step by step (e.g., <think>Reasoning process here</think>). Provide your final answer as the selected multiple-choice option (e.g., Option A, Option B) and enclose it within <answer> tags (e.g., <answer>Option A</answer>).",
                                    "Carefully read the provided text, which narrates a story or sequence of events. Use your vision-language reasoning skills to analyze and reorder the images so they logically align with the narrative structure. As you reason through the process, enclose your step-by-step thinking within <think> and </think> tags. Once you identify the correct sequence of images from the multiple-choice options (e.g., Option A, Option B), enclose your final answer within <answer> and </answer> tags.",
                                    "Order the following images in the correct sequence based on the content of the story. Compare each image with the text description, carefully analyzing the sequence of events to determine the proper order of the unordered images. \nResponse shoould be in this format:\n<think> First, examine the text description to identify key events and their chronological order. Next, analyze each image to match it with the corresponding event described in the text. Consider visual cues, actions, and details in the images that indicate the progression of the story. Arrange the images accordingly to reflect the correct sequence of events. </think>\n<answer> Option [Your Choice] </answer>"
                                    ]
    
    event_columns = [col for col in df.columns if col.startswith('event')]
    text_columns = [col for col in df.columns if col.startswith('text')]
    
    if task_name == 'paired_event_discrimination':
        column_names = ["data_source", "img_seq_id", "question_id", "task_name", "event_pair_tuple", "input_image_type", "event_distance",
                        "event_pre", "event_post", "text_pre", "text_post", "img_pre", "img_post", "img_path_list", "combined_img_path",
                        "statement_type", "statement_text", "task_instruction", "option_text", "gt_option", "character_info",
                        "blind_instruction"
                        ]
        task_instruction_list = task_instructions[task_name]
        requirement_template_list = requirement_templates[task_name]
        answer_format_template_list = answer_format_templates[task_name]
        full_instructions = generate_combinations(task_instruction_list, requirement_template_list, answer_format_template_list)
        question_df = pd.DataFrame(columns=column_names)
        global_index = 0
        for index, row in df.iterrows():
            available_pairs = eval(row['available_pairs']) 
            for pair in available_pairs:
                for q_type in ['gt', 'nc']:
                    question_df.at[global_index, 'data_source'] = data_source
                    if data_source in ['flintstones', 'pororo']:
                        question_df.at[global_index, 'img_seq_id'] = row['img_seq_id']
                        question_df.at[global_index, 'question_id'] = f'{row['img_seq_id']}_q{global_index}'
                    elif data_source in ['vwp']:
                        question_df.at[global_index, 'img_seq_id'] = row['scene_full_id']
                        question_df.at[global_index, 'question_id'] = f'{row['scene_full_id']}_q{global_index}'
                    else: #data_source in ['vist']
                        question_df.at[global_index, 'img_seq_id'] = row['album_id']
                        question_df.at[global_index, 'question_id'] = f'{row['story_id']}_q{global_index}'

                    question_df.at[global_index, 'task_name'] = task_name
                    question_df.at[global_index, 'event_pair_tuple'] = pair
                    question_df.at[global_index, 'event_distance'] = pair[1] - pair[0]
                    question_df.at[global_index, 'input_image_type'] = 'combined'
                    question_df.at[global_index, 'event_pre'] = row[f"event{pair[0]}"]
                    question_df.at[global_index, 'event_post'] = row[f"event{pair[1]}"]
                    question_df.at[global_index, 'text_pre'] = row[f"text{pair[0]}"]
                    question_df.at[global_index, 'text_post'] = row[f"text{pair[1]}"]
                    question_df.at[global_index, 'img_pre'] = row[f"link{pair[0]}"]
                    question_df.at[global_index, 'img_post'] = row[f"link{pair[1]}"]
                    img_path_list = [row[col] for col in row.index if col.startswith('link')]
                    img_path_list = [item for item in img_path_list if not (isinstance(item, float) and math.isnan(item))]
                    if not validate_image_paths(img_path_list):
                        continue
                    question_df.at[global_index, 'img_path_list'] = img_path_list
                    combined_img_path =  get_combined_img_path(data_source, f'/data/{data_source}', img_path_list)
                    question_df.at[global_index, 'combined_img_path'] = combined_img_path
                    if not os.path.exists(combined_img_path):
                        combine_image_seq(data_source, f'/data/{data_source}', img_path_list, use_number=True)
                    statement_type = statement_types_paired[global_index % len(statement_types_paired)] + f'_{q_type}'
                    question_df.at[global_index, 'statement_type'] = statement_type
                    question_df.at[global_index, 'statement_text'] = create_statements_pair(row[f"event{pair[0]}"], row[f"event{pair[1]}"], statement_type)
                    instruction_index = global_index % len(full_instructions)
                    question_df.at[global_index, 'task_instruction'] = full_instructions[instruction_index]
                    cot_instruction_index = global_index % len(statement_instruction_cot)
                    question_df.at[global_index, 'task_instruction_cot'] = statement_instruction_cot[cot_instruction_index]
                    option_index = global_index % len(option_templates[task_name])
                    question_df.at[global_index, 'option_text'] = option_templates[task_name][option_index]
                    question_df.at[global_index, 'gt_option'] = get_gt_answer_statement(statement_type, task_name, option_index)
                    question_df.at[global_index, 'character_info'] = " ".join(list(set(find_characters_in_string(row[f"event{pair[0]}"], data_source) + find_characters_in_string(row[f"event{pair[1]}"], data_source))))
                    blind_index = global_index % len(blind_instruction_statement)
                    question_df.at[global_index, 'blind_instruction'] = blind_instruction_statement[blind_index]
                    # question_df.at[global_index, 'blind_instruction_reason'] = blind_instruction_statement_reason
                    global_index += 1
                    
                    
    elif task_name == 'reversed_paired_event_discrimination':
        column_names = ["data_source", "img_seq_id", "question_id", "task_name", "event_pair_tuple", "input_image_type", "event_distance",
                        "event_pre", "event_post", "text_pre", "text_post", "img_pre", "img_post", "img_path_list_reversed", "combined_img_path_reversed",
                        "statement_type", "statement_text", "task_instruction", "option_text", "gt_option", "character_info",
                        "blind_instruction"
                        ]
        task_instruction_list = task_instructions[task_name]
        requirement_template_list = requirement_templates[task_name]
        answer_format_template_list = answer_format_templates[task_name]
        full_instructions = generate_combinations(task_instruction_list, requirement_template_list, answer_format_template_list)
        question_df = pd.DataFrame(columns=column_names)
        global_index = 0
        for index, row in df.iterrows():
            available_pairs = eval(row['available_pairs']) 
            for pair in available_pairs:
                for q_type in ['gt', 'nc']:
                    question_df.at[global_index, 'data_source'] = data_source
                    if data_source in ['flintstones', 'pororo']:
                        question_df.at[global_index, 'img_seq_id'] = row['img_seq_id']
                        question_df.at[global_index, 'question_id'] = f'{row['img_seq_id']}_q{global_index}'
                    elif data_source in ['vwp']:
                        question_df.at[global_index, 'img_seq_id'] = row['scene_full_id']
                        question_df.at[global_index, 'question_id'] = f'{row['scene_full_id']}_q{global_index}'
                    else: #data_source in ['vist']
                        question_df.at[global_index, 'img_seq_id'] = row['album_id']
                        question_df.at[global_index, 'question_id'] = f'{row['story_id']}_q{global_index}'

                    question_df.at[global_index, 'task_name'] = task_name
                    question_df.at[global_index, 'event_pair_tuple'] = pair
                    question_df.at[global_index, 'event_distance'] = pair[1] - pair[0]
                    question_df.at[global_index, 'input_image_type'] = 'combined'
                    question_df.at[global_index, 'event_pre'] = row[f"event{pair[0]}"]
                    question_df.at[global_index, 'event_post'] = row[f"event{pair[1]}"]
                    question_df.at[global_index, 'text_pre'] = row[f"text{pair[0]}"]
                    question_df.at[global_index, 'text_post'] = row[f"text{pair[1]}"]
                    question_df.at[global_index, 'img_pre'] = row[f"link{pair[0]}"]
                    question_df.at[global_index, 'img_post'] = row[f"link{pair[1]}"]
                    img_path_list = [row[col] for col in row.index if col.startswith('link')]
                    img_path_list = [item for item in img_path_list if not (isinstance(item, float) and math.isnan(item))]
                    if not validate_image_paths(img_path_list):
                        continue
                    
                    question_df.at[global_index, 'img_path_list_reversed'] = img_path_list[::-1]
                    combined_img_path_reversed = get_combined_img_path(data_source, f'/data/{data_source}', img_path_list[::-1], use_number=True, reversed=True)
                    question_df.at[global_index, 'combined_img_path_reversed'] = combined_img_path_reversed
                    if not os.path.exists(combined_img_path_reversed):
                        combine_image_seq(data_source, f'/data/{data_source}', img_path_list[::-1], use_number=True, reversed=True)
            
                    
                    
                    
                    statement_type = statement_types_paired[global_index % len(statement_types_paired)] + f'_{q_type}'
                    question_df.at[global_index, 'statement_type'] = statement_type
                    question_df.at[global_index, 'statement_text'] = create_statements_pair(row[f"event{pair[0]}"], row[f"event{pair[1]}"], statement_type)
                    instruction_index = global_index % len(full_instructions)
                    question_df.at[global_index, 'task_instruction'] = full_instructions[instruction_index]
                    cot_instruction_index = global_index % len(statement_instruction_cot)
                    question_df.at[global_index, 'task_instruction_cot'] = statement_instruction_cot[cot_instruction_index]
                    option_index = global_index % len(option_templates['paired_event_discrimination'])
                    question_df.at[global_index, 'option_text'] = option_templates['paired_event_discrimination'][option_index]
                    question_df.at[global_index, 'gt_option'] = get_gt_answer_statement(statement_type, 'paired_event_discrimination', option_index)
                    question_df.at[global_index, 'character_info'] = " ".join(list(set(find_characters_in_string(row[f"event{pair[0]}"], data_source) + find_characters_in_string(row[f"event{pair[1]}"], data_source))))
                    blind_index = global_index % len(blind_instruction_statement)
                    question_df.at[global_index, 'blind_instruction'] = blind_instruction_statement[blind_index]
                    # question_df.at[global_index, 'blind_instruction_reason'] = blind_instruction_statement_reason
                    global_index += 1
                    
    elif task_name == 'triple_event_discrimination':
        column_names = ["data_source", "img_seq_id", "question_id", "task_name", "event_triple_tuple", "input_image_type", 
                        "event_pre", "event_mid", "event_post", "text_pre", "text_mid", "text_post", "img_pre", "img_mid", "img_post", 
                        "img_path_list", "combined_img_path",
                        "statement_type", "statement_text", "task_instruction", "option_text", "gt_option", "character_info",
                        "blind_instruction"]
        task_instruction_list = task_instructions['paired_event_discrimination']
        requirement_template_list = requirement_templates['paired_event_discrimination']
        answer_format_template_list = answer_format_templates['paired_event_discrimination']
        full_instructions = generate_combinations(task_instruction_list, requirement_template_list, answer_format_template_list)
        question_df = pd.DataFrame(columns=column_names)
        global_index = 0
        for index, row in df.iterrows():
            available_triples = row['available_triples']
            for triple in available_triples:
                for q_type in ['gt', 'nc']:
                    question_df.at[global_index, 'data_source'] = data_source
                    if data_source in ['flintstones', 'pororo']:
                        question_df.at[global_index, 'img_seq_id'] = row['img_seq_id']
                        question_df.at[global_index, 'question_id'] = f'{row['img_seq_id']}_q{global_index}'
                    elif data_source in ['vwp']:
                        question_df.at[global_index, 'img_seq_id'] = row['scene_full_id']
                        question_df.at[global_index, 'question_id'] = f'{row['scene_full_id']}_q{global_index}'
                    else: #data_source in ['vist']
                        question_df.at[global_index, 'img_seq_id'] = row['album_id']
                        question_df.at[global_index, 'question_id'] = f'{row['story_id']}_q{global_index}'

                    question_df.at[global_index, 'task_name'] = task_name
                    question_df.at[global_index, 'event_triple_tuple'] = triple
                    question_df.at[global_index, 'input_image_type'] = 'combined'
                    question_df.at[global_index, 'event_pre'] = row[f"event{triple[0]}"]
                    question_df.at[global_index, 'event_mid'] = row[f"event{triple[1]}"]
                    question_df.at[global_index, 'event_post'] = row[f"event{triple[2]}"]
                    question_df.at[global_index, 'text_pre'] = row[f"text{triple[0]}"]
                    question_df.at[global_index, 'text_mid'] = row[f"text{triple[1]}"]
                    question_df.at[global_index, 'text_post'] = row[f"text{triple[2]}"]
                    question_df.at[global_index, 'img_pre'] = row[f"link{triple[0]}"]
                    question_df.at[global_index, 'img_mid'] = row[f"link{triple[1]}"]
                    question_df.at[global_index, 'img_post'] = row[f"link{triple[2]}"]
                    img_path_list = [row[col] for col in row.index if col.startswith('link')]
                    img_path_list = [item for item in img_path_list if not (isinstance(item, float) and math.isnan(item))]
                    if not validate_image_paths(img_path_list):
                        continue
                    question_df.at[global_index, 'img_path_list'] = img_path_list
                    combined_img_path =  get_combined_img_path(data_source, f'/data/{data_source}', img_path_list)
                    question_df.at[global_index, 'combined_img_path'] = combined_img_path
                    if not os.path.exists(combined_img_path):
                        combine_image_seq(data_source, f'/data/{data_source}', img_path_list, use_number=True)
                    statement_type = statement_types_triple[global_index % len(statement_types_triple)] + f'_{q_type}'
                    question_df.at[global_index, 'statement_type'] = statement_type
                    question_df.at[global_index, 'statement_text'] = create_statements_triple(row[f"event{triple[0]}"], row[f"event{triple[1]}"], row[f"event{triple[2]}"], statement_type, seed_value=global_index)
                    instruction_index = global_index % len(full_instructions)
                    question_df.at[global_index, 'task_instruction'] = full_instructions[instruction_index]
                    cot_instruction_index = global_index % len(statement_instruction_cot)
                    question_df.at[global_index, 'task_instruction_cot'] = statement_instruction_cot[cot_instruction_index]
                    option_index = global_index % len(option_templates[task_name])
                    question_df.at[global_index, 'option_text'] = option_templates[task_name][option_index]
                    question_df.at[global_index, 'gt_option'] = get_gt_answer_statement(statement_type, task_name, option_index)
                    question_df.at[global_index, 'character_info'] = " ".join(list(set(find_characters_in_string(row[f"event{triple[0]}"], data_source) + find_characters_in_string(row[f"event{triple[1]}"], data_source)+ find_characters_in_string(row[f"event{triple[2]}"], data_source))))
                    blind_index = global_index % len(blind_instruction_statement)
                    question_df.at[global_index, 'blind_instruction'] = blind_instruction_statement[blind_index]
                    # question_df.at[global_index, 'blind_instruction_reason'] = blind_instruction_statement_reason
                    global_index += 1
                    
    elif task_name == 'reversed_triple_event_discrimination':
        column_names = ["data_source", "img_seq_id", "question_id", "task_name", "event_triple_tuple", "input_image_type", 
                        "event_pre", "event_mid", "event_post", "text_pre", "text_mid", "text_post", "img_pre", "img_mid", "img_post", 
                        "img_path_list_reversed", "combined_img_path_reversed",
                        "statement_type", "statement_text", "task_instruction", "option_text", "gt_option", "character_info",
                        "blind_instruction"]
        task_instruction_list = task_instructions['reversed_paired_event_discrimination']
        requirement_template_list = requirement_templates['reversed_paired_event_discrimination']
        answer_format_template_list = answer_format_templates['reversed_paired_event_discrimination']
        full_instructions = generate_combinations(task_instruction_list, requirement_template_list, answer_format_template_list)
        question_df = pd.DataFrame(columns=column_names)
        global_index = 0
        for index, row in df.iterrows():
            available_triples = row['available_triples']
            for triple in available_triples:
                for q_type in ['gt', 'nc']:
                    question_df.at[global_index, 'data_source'] = data_source
                    if data_source in ['flintstones', 'pororo']:
                        question_df.at[global_index, 'img_seq_id'] = row['img_seq_id']
                        question_df.at[global_index, 'question_id'] = f'{row['img_seq_id']}_q{global_index}'
                    elif data_source in ['vwp']:
                        question_df.at[global_index, 'img_seq_id'] = row['scene_full_id']
                        question_df.at[global_index, 'question_id'] = f'{row['scene_full_id']}_q{global_index}'
                    else: #data_source in ['vist']
                        question_df.at[global_index, 'img_seq_id'] = row['album_id']
                        question_df.at[global_index, 'question_id'] = f'{row['story_id']}_q{global_index}'

                    question_df.at[global_index, 'task_name'] = task_name
                    question_df.at[global_index, 'event_triple_tuple'] = triple
                    question_df.at[global_index, 'input_image_type'] = 'combined'
                    question_df.at[global_index, 'event_pre'] = row[f"event{triple[0]}"]
                    question_df.at[global_index, 'event_mid'] = row[f"event{triple[1]}"]
                    question_df.at[global_index, 'event_post'] = row[f"event{triple[2]}"]
                    question_df.at[global_index, 'text_pre'] = row[f"text{triple[0]}"]
                    question_df.at[global_index, 'text_mid'] = row[f"text{triple[1]}"]
                    question_df.at[global_index, 'text_post'] = row[f"text{triple[2]}"]
                    question_df.at[global_index, 'img_pre'] = row[f"link{triple[0]}"]
                    question_df.at[global_index, 'img_mid'] = row[f"link{triple[1]}"]
                    question_df.at[global_index, 'img_post'] = row[f"link{triple[2]}"]
                    img_path_list = [row[col] for col in row.index if col.startswith('link')]
                    img_path_list = [item for item in img_path_list if not (isinstance(item, float) and math.isnan(item))]
                    if not validate_image_paths(img_path_list):
                        continue
                    question_df.at[global_index, 'img_path_list_reversed'] = img_path_list[::-1]
                    combined_img_path_reversed = get_combined_img_path(data_source, f'/data/{data_source}', img_path_list[::-1], use_number=True, reversed=True)
                    question_df.at[global_index, 'combined_img_path_reversed'] = combined_img_path_reversed
                    if not os.path.exists(combined_img_path_reversed):
                        combine_image_seq(data_source, f'/data/{data_source}', img_path_list[::-1], use_number=True, reversed=True)
            
                    
                    
                    statement_type = statement_types_triple[global_index % len(statement_types_triple)] + f'_{q_type}'
                    question_df.at[global_index, 'statement_type'] = statement_type
                    question_df.at[global_index, 'statement_text'] = create_statements_triple(row[f"event{triple[0]}"], row[f"event{triple[1]}"], row[f"event{triple[2]}"], statement_type, seed_value=global_index)
                    instruction_index = global_index % len(full_instructions)
                    question_df.at[global_index, 'task_instruction'] = full_instructions[instruction_index]
                    cot_instruction_index = global_index % len(statement_instruction_cot)
                    question_df.at[global_index, 'task_instruction_cot'] = statement_instruction_cot[cot_instruction_index]
                    option_index = global_index % len(option_templates['triple_event_discrimination'])
                    question_df.at[global_index, 'option_text'] = option_templates['triple_event_discrimination'][option_index]
                    question_df.at[global_index, 'gt_option'] = get_gt_answer_statement(statement_type, 'triple_event_discrimination', option_index)
                    question_df.at[global_index, 'character_info'] = " ".join(list(set(find_characters_in_string(row[f"event{triple[0]}"], data_source) + find_characters_in_string(row[f"event{triple[1]}"], data_source)+ find_characters_in_string(row[f"event{triple[2]}"], data_source))))
                    blind_index = global_index % len(blind_instruction_statement)
                    question_df.at[global_index, 'blind_instruction'] = blind_instruction_statement[blind_index]
                    # question_df.at[global_index, 'blind_instruction_reason'] = blind_instruction_statement_reason
                    global_index += 1
                    
    elif task_name == 'paired_grounding_one_text':
        column_names = ["data_source", "img_seq_id", "question_id", "task_name", "event_pair_tuple", "input_image_type", "event_distance",
                        "event_pre", "event_post", "text_pre", "text_post", "img_pre", "img_post", "img_path_list", "combined_img_path",
                        "event_to_test","text_to_test", "task_instruction", "option_text", "gt_option", "character_info"]
        task_instruction_list = task_instructions[task_name]
        requirement_template_list = requirement_templates[task_name]
        answer_format_template_list = answer_format_templates['paired_event_discrimination']
        full_instructions = generate_combinations(task_instruction_list, requirement_template_list, answer_format_template_list)
        question_df = pd.DataFrame(columns=column_names)
        global_index = 0
        for index, row in df.iterrows():
            available_pairs = eval(row['available_pairs'])
            #event_indices = sorted(set(num for tup in available_pairs for num in tup))
            for pair in available_pairs:
                for ind_tup in [0, 1]:
                    event_index = pair[ind_tup]
                    question_df.at[global_index, 'data_source'] = data_source
                    if data_source in ['flintstones', 'pororo']:
                        question_df.at[global_index, 'img_seq_id'] = row['img_seq_id']
                        question_df.at[global_index, 'question_id'] = f'{row['img_seq_id']}_q{global_index}'
                    elif data_source in ['vwp']:
                        question_df.at[global_index, 'img_seq_id'] = row['scene_full_id']
                        question_df.at[global_index, 'question_id'] = f'{row['scene_full_id']}_q{global_index}'
                    else: #data_source in ['vist']
                        question_df.at[global_index, 'img_seq_id'] = row['album_id']
                        question_df.at[global_index, 'question_id'] = f'{row['story_id']}_q{global_index}'

                    question_df.at[global_index, 'task_name'] = task_name
                    question_df.at[global_index, 'event_pair_tuple'] = pair
                    question_df.at[global_index, 'event_distance'] = pair[1] - pair[0]
                    question_df.at[global_index, 'input_image_type'] = 'separate'
                    question_df.at[global_index, 'event_pre'] = row[f"event{pair[0]}"]
                    question_df.at[global_index, 'event_post'] = row[f"event{pair[1]}"]
                    question_df.at[global_index, 'text_pre'] = row[f"text{pair[0]}"]
                    question_df.at[global_index, 'text_post'] = row[f"text{pair[1]}"]
                    question_df.at[global_index, 'img_pre'] = row[f"link{pair[0]}"]
                    question_df.at[global_index, 'img_post'] = row[f"link{pair[1]}"]
                    question_df.at[global_index, 'event_to_test'] = row[f"event{event_index}"]
                    question_df.at[global_index, 'text_to_test'] = row[f"text{event_index}"]
                    img_path_list = [row[f"link{pair[0]}"], row[f"link{pair[1]}"]]
                    if not validate_image_paths(img_path_list):
                        continue
                    question_df.at[global_index, 'img_path_list'] = img_path_list
                    combined_img_path =  get_combined_img_path(data_source, f'/data/{data_source}', img_path_list)
                    question_df.at[global_index, 'combined_img_path'] = combined_img_path
                    instruction_index = global_index % len(full_instructions)
                    question_df.at[global_index, 'task_instruction'] = full_instructions[instruction_index]
                    option_index = global_index % len(option_templates[task_name])
                    question_df.at[global_index, 'option_text'] = option_templates[task_name][option_index]
                    question_df.at[global_index, 'gt_option'] = get_gt_answer_paired_triple_grounding(ind_tup, task_name, option_index)
                    question_df.at[global_index, 'character_info'] = " ".join(list(set(find_characters_in_string(row[f"text{pair[0]}"], data_source) + find_characters_in_string(row[f"text{pair[1]}"], data_source))))
                    global_index += 1
                    
    elif task_name == 'triple_grounding_one_text':
        column_names = ["data_source", "img_seq_id", "question_id", "task_name", "event_triple_tuple", "input_image_type", 
                        "event_pre", "event_mid", "event_post", "text_pre", "text_mid", "text_post", "img_pre", "img_mid", "img_post", 
                        "img_path_list", "combined_img_path",
                        "event_to_test","text_to_test", "task_instruction", "option_text", "gt_option", "character_info"]
        task_instruction_list = task_instructions["paired_grounding_one_text"]
        requirement_template_list = requirement_templates[task_name]
        answer_format_template_list = answer_format_templates['paired_event_discrimination']
        full_instructions = generate_combinations(task_instruction_list, requirement_template_list, answer_format_template_list)
        question_df = pd.DataFrame(columns=column_names)
        global_index = 0
        for index, row in df.iterrows():
            available_triples = row['available_triples']
            #event_indices = sorted(set(num for tup in available_pairs for num in tup))
            for triple in available_triples:
                for ind_tup in [0, 1, 2]:
                    event_index = triple[ind_tup]
                    question_df.at[global_index, 'data_source'] = data_source
                    if data_source in ['flintstones', 'pororo']:
                        question_df.at[global_index, 'img_seq_id'] = row['img_seq_id']
                        question_df.at[global_index, 'question_id'] = f'{row['img_seq_id']}_q{global_index}'
                    elif data_source in ['vwp']:
                        question_df.at[global_index, 'img_seq_id'] = row['scene_full_id']
                        question_df.at[global_index, 'question_id'] = f'{row['scene_full_id']}_q{global_index}'
                    else: #data_source in ['vist']
                        question_df.at[global_index, 'img_seq_id'] = row['album_id']
                        question_df.at[global_index, 'question_id'] = f'{row['story_id']}_q{global_index}'

                    question_df.at[global_index, 'task_name'] = task_name
                    question_df.at[global_index, 'event_triple_tuple'] = triple
                    question_df.at[global_index, 'input_image_type'] = 'separate'
                    question_df.at[global_index, 'event_pre'] = row[f"event{triple[0]}"]
                    question_df.at[global_index, 'event_mid'] = row[f"event{triple[1]}"]
                    question_df.at[global_index, 'event_post'] = row[f"event{triple[2]}"]
                    question_df.at[global_index, 'text_pre'] = row[f"text{triple[0]}"]
                    question_df.at[global_index, 'text_mid'] = row[f"text{triple[1]}"]
                    question_df.at[global_index, 'text_post'] = row[f"text{triple[2]}"]
                    question_df.at[global_index, 'img_pre'] = row[f"link{triple[0]}"]
                    question_df.at[global_index, 'img_mid'] = row[f"link{triple[1]}"]
                    question_df.at[global_index, 'img_post'] = row[f"link{triple[2]}"]
                    question_df.at[global_index, 'event_to_test'] = row[f"event{event_index}"]
                    question_df.at[global_index, 'text_to_test'] = row[f"text{event_index}"]
                    img_path_list = [row[f"link{triple[0]}"], row[f"link{triple[1]}"], row[f"link{triple[2]}"]]
                    if not validate_image_paths(img_path_list):
                        continue
                    question_df.at[global_index, 'img_path_list'] = img_path_list
                    combined_img_path =  get_combined_img_path(data_source, f'/data/{data_source}', img_path_list)
                    question_df.at[global_index, 'combined_img_path'] = combined_img_path
                    instruction_index = global_index % len(full_instructions)
                    question_df.at[global_index, 'task_instruction'] = full_instructions[instruction_index]
                    option_index = global_index % len(option_templates[task_name])
                    question_df.at[global_index, 'option_text'] = option_templates[task_name][option_index]
                    question_df.at[global_index, 'gt_option'] = get_gt_answer_paired_triple_grounding(ind_tup, task_name, option_index)
                    question_df.at[global_index, 'character_info'] = " ".join(list(set(find_characters_in_string(row[f"text{triple[0]}"], data_source) + find_characters_in_string(row[f"text{triple[1]}"], data_source) + find_characters_in_string(row[f"text{triple[2]}"], data_source))))
                    global_index += 1
    
    elif task_name == 'paired_grounding_one_image':
        column_names = ["data_source", "img_seq_id", "question_id", "task_name", "event_pair_tuple", "input_image_type", "event_distance",
                        "event_pre", "event_post", "text_pre", "text_post", "img_pre", "img_post", "img_path_list", 
                        "task_instruction", "option_text", "gt_option", "character_info"]
        
        task_instruction_list = task_instructions[task_name]
        requirement_template_list = requirement_templates[task_name]
        answer_format_template_list = answer_format_templates['paired_event_discrimination']
        full_instructions = generate_combinations(task_instruction_list, requirement_template_list, answer_format_template_list)
        question_df = pd.DataFrame(columns=column_names)
        global_index = 0
        for index, row in df.iterrows():
            available_pairs = eval(row['available_pairs'])
            #event_indices = sorted(set(num for tup in available_pairs for num in tup))
            for pair in available_pairs:
                for ind_tup in [0, 1]:
                    event_index = pair[ind_tup]
                    question_df.at[global_index, 'data_source'] = data_source
                    if data_source in ['flintstones', 'pororo']:
                        question_df.at[global_index, 'img_seq_id'] = row['img_seq_id']
                        question_df.at[global_index, 'question_id'] = f'{row['img_seq_id']}_q{global_index}'
                    elif data_source in ['vwp']:
                        question_df.at[global_index, 'img_seq_id'] = row['scene_full_id']
                        question_df.at[global_index, 'question_id'] = f'{row['scene_full_id']}_q{global_index}'
                    else: #data_source in ['vist']
                        question_df.at[global_index, 'img_seq_id'] = row['album_id']
                        question_df.at[global_index, 'question_id'] = f'{row['story_id']}_q{global_index}'

                    question_df.at[global_index, 'task_name'] = task_name
                    question_df.at[global_index, 'event_pair_tuple'] = pair
                    question_df.at[global_index, 'event_distance'] = pair[1] - pair[0]
                    question_df.at[global_index, 'input_image_type'] = 'separate'
                    question_df.at[global_index, 'event_pre'] = row[f"event{pair[0]}"]
                    question_df.at[global_index, 'event_post'] = row[f"event{pair[1]}"]
                    question_df.at[global_index, 'text_pre'] = row[f"text{pair[0]}"]
                    question_df.at[global_index, 'text_post'] = row[f"text{pair[1]}"]
                    question_df.at[global_index, 'img_pre'] = row[f"link{pair[0]}"]
                    question_df.at[global_index, 'img_post'] = row[f"link{pair[1]}"]
                    question_df.at[global_index, 'img_path_list'] = [row[f"link{pair[0]}"], row[f"link{pair[1]}"]]
                    question_df.at[global_index, 'image_to_test'] = row[f"link{event_index}"]
                    instruction_index = global_index % len(full_instructions)
                    question_df.at[global_index, 'task_instruction'] = full_instructions[instruction_index]
                    option_index = global_index % len(option_templates[task_name])
                    question_df.at[global_index, 'option_text'] = option_templates[task_name][option_index]
                    question_df.at[global_index, 'gt_option'] = get_gt_answer_paired_triple_grounding(ind_tup, task_name, option_index)
                    question_df.at[global_index, 'character_info'] = " ".join(list(set(find_characters_in_string(row[f"text{pair[0]}"], data_source) + find_characters_in_string(row[f"text{pair[1]}"], data_source))))

                    global_index += 1
                    
    elif task_name == 'triple_grounding_one_image':
        column_names = ["data_source", "img_seq_id", "question_id", "task_name", "event_triple_tuple", "input_image_type", "event_distance",
                        "event_pre", "event_mid", "event_post", "text_pre", "text_mid", "text_post", "img_pre", "img_mid", "img_post",  
                        "task_instruction", "option_text", "gt_option", "character_info"]
        task_instruction_list = task_instructions["paired_grounding_one_image"]
        requirement_template_list = requirement_templates[task_name]
        answer_format_template_list = answer_format_templates['paired_event_discrimination']
        full_instructions = generate_combinations(task_instruction_list, requirement_template_list, answer_format_template_list)
        
        question_df = pd.DataFrame(columns=column_names)
        global_index = 0
        for index, row in df.iterrows():
            available_triples = row['available_triples']
            for triple in available_triples:
                for ind_tup in [0, 1, 2]:
                    event_index = triple[ind_tup]
                    question_df.at[global_index, 'data_source'] = data_source
                    if data_source in ['flintstones', 'pororo']:
                        question_df.at[global_index, 'img_seq_id'] = row['img_seq_id']
                        question_df.at[global_index, 'question_id'] = f'{row['img_seq_id']}_q{global_index}'
                    elif data_source in ['vwp']:
                        question_df.at[global_index, 'img_seq_id'] = row['scene_full_id']
                        question_df.at[global_index, 'question_id'] = f'{row['scene_full_id']}_q{global_index}'
                    else: #data_source in ['vist']
                        question_df.at[global_index, 'img_seq_id'] = row['album_id']
                        question_df.at[global_index, 'question_id'] = f'{row['story_id']}_q{global_index}'

                    question_df.at[global_index, 'task_name'] = task_name
                    question_df.at[global_index, 'event_triple_tuple'] = triple
                    question_df.at[global_index, 'input_image_type'] = 'separate'
                    question_df.at[global_index, 'event_pre'] = row[f"event{triple[0]}"]
                    question_df.at[global_index, 'event_mid'] = row[f"event{triple[1]}"]
                    question_df.at[global_index, 'event_post'] = row[f"event{triple[2]}"]
                    question_df.at[global_index, 'text_pre'] = row[f"text{triple[0]}"]
                    question_df.at[global_index, 'text_mid'] = row[f"text{triple[1]}"]
                    question_df.at[global_index, 'text_post'] = row[f"text{triple[2]}"]
                    question_df.at[global_index, 'img_pre'] = row[f"link{triple[0]}"]
                    question_df.at[global_index, 'img_mid'] = row[f"link{triple[1]}"]
                    question_df.at[global_index, 'img_post'] = row[f"link{triple[2]}"]
                    question_df.at[global_index, 'image_to_test'] = row[f"link{event_index}"]
                    instruction_index = global_index % len(full_instructions)
                    question_df.at[global_index, 'task_instruction'] = full_instructions[instruction_index]
                    option_index = global_index % len(option_templates[task_name])
                    question_df.at[global_index, 'option_text'] = option_templates[task_name][option_index]
                    question_df.at[global_index, 'gt_option'] = get_gt_answer_paired_triple_grounding(ind_tup, task_name, option_index)
                    question_df.at[global_index, 'character_info'] = " ".join(list(set(find_characters_in_string(row[f"text{triple[0]}"], data_source) + find_characters_in_string(row[f"text{triple[1]}"], data_source) + find_characters_in_string(row[f"text{triple[2]}"], data_source))))

                    global_index += 1



    elif task_name == 'single_grounding_paired':
        df['all_events'] = df.filter(regex='^event').apply(lambda row: ''.join(row.values.astype(str)), axis=1)
        column_names = ["data_source", "img_seq_id", "question_id", "task_name", "input_image_type", 
                        "img_path_list", "combined_img_path",
                        "event_to_test","text_to_test", "task_instruction", "gt_option", "character_info"]
        task_instruction_list = task_instructions[task_name[:-7]]
        requirement_template_list = requirement_templates['paired_event_discrimination']
        answer_format_template_list = answer_format_templates[task_name[:-7]]
        full_instructions = generate_combinations(task_instruction_list, requirement_template_list, answer_format_template_list)
        question_df = pd.DataFrame(columns=column_names)
        global_index = 0
        for index, row in df.iterrows():
            available_pairs = eval(row['available_pairs'])
            event_indices = sorted(set(num for tup in available_pairs for num in tup))
            for event_index in event_indices:
                question_df.at[global_index, 'data_source'] = data_source
                if data_source in ['flintstones', 'pororo']:
                    question_df.at[global_index, 'img_seq_id'] = row['img_seq_id']
                    question_df.at[global_index, 'question_id'] = f'{row['img_seq_id']}_q{global_index}'
                elif data_source in ['vwp']:
                    question_df.at[global_index, 'img_seq_id'] = row['scene_full_id']
                    question_df.at[global_index, 'question_id'] = f'{row['scene_full_id']}_q{global_index}'
                else: #data_source in ['vist']
                        question_df.at[global_index, 'img_seq_id'] = row['album_id']
                        question_df.at[global_index, 'question_id'] = f'{row['story_id']}_q{global_index}'
                        
                question_df.at[global_index, 'task_name'] = task_name
                question_df.at[global_index, 'input_image_type'] = 'combined'
                question_df.at[global_index, 'event_to_test'] = row[f"event{event_index}"]
                question_df.at[global_index, 'text_to_test'] = row[f"text{event_index}"]
                question_df.at[global_index, 'img_to_test'] = row[f"link{event_index}"]
                img_path_list = [row[col] for col in row.index if col.startswith('link')]
                img_path_list = [item for item in img_path_list if not (isinstance(item, float) and math.isnan(item))]
                if not validate_image_paths(img_path_list):
                    continue
                question_df.at[global_index, 'img_path_list'] = img_path_list
                combined_img_path =  get_combined_img_path(data_source, f'/data/{data_source}', img_path_list)
                question_df.at[global_index, 'combined_img_path'] = combined_img_path
                if not os.path.exists(combined_img_path):
                    combine_image_seq(data_source, f'/data/{data_source}', img_path_list, use_number=True)
                instruction_index = global_index % len(full_instructions)
                question_df.at[global_index, 'task_instruction'] = full_instructions[instruction_index]
                question_df.at[global_index, 'gt_option'] = f'image {event_index + 1}'
                question_df.at[global_index, 'character_info'] = " ".join(list(set(find_characters_in_string(row['all_events'], data_source) )))
                global_index += 1
                
    elif task_name == 'single_grounding_triple':
        df['all_events'] = df.filter(regex='^event').apply(lambda row: ''.join(row.values.astype(str)), axis=1)
        column_names = ["data_source", "img_seq_id", "question_id", "task_name", "input_image_type", 
                        "img_path_list", "combined_img_path",
                        "event_to_test","text_to_test", "task_instruction", "gt_option", "character_info"]
        task_instruction_list = task_instructions[task_name[:-7]]
        requirement_template_list = requirement_templates['paired_event_discrimination']
        answer_format_template_list = answer_format_templates[task_name[:-7]]
        full_instructions = generate_combinations(task_instruction_list, requirement_template_list, answer_format_template_list)
        question_df = pd.DataFrame(columns=column_names)
        global_index = 0
        for index, row in df.iterrows():
            available_triples = row['available_triples']
            event_indices = sorted(set(num for tup in available_triples for num in tup))
            for event_index in event_indices:
                question_df.at[global_index, 'data_source'] = data_source
                if data_source in ['flintstones', 'pororo']:
                    question_df.at[global_index, 'img_seq_id'] = row['img_seq_id']
                    question_df.at[global_index, 'question_id'] = f'{row['img_seq_id']}_q{global_index}'
                elif data_source in ['vwp']:
                    question_df.at[global_index, 'img_seq_id'] = row['scene_full_id']
                    question_df.at[global_index, 'question_id'] = f'{row['scene_full_id']}_q{global_index}'
                else: #data_source in ['vist']
                        question_df.at[global_index, 'img_seq_id'] = row['album_id']
                        question_df.at[global_index, 'question_id'] = f'{row['story_id']}_q{global_index}'
                        
                question_df.at[global_index, 'task_name'] = task_name
                question_df.at[global_index, 'input_image_type'] = 'combined'
                question_df.at[global_index, 'event_to_test'] = row[f"event{event_index}"]
                question_df.at[global_index, 'text_to_test'] = row[f"text{event_index}"]
                question_df.at[global_index, 'img_to_test'] = row[f"link{event_index}"]
                img_path_list = [row[col] for col in row.index if col.startswith('link')]
                img_path_list = [item for item in img_path_list if not (isinstance(item, float) and math.isnan(item))]
                if not validate_image_paths(img_path_list):
                    continue
                question_df.at[global_index, 'img_path_list'] = img_path_list
                combined_img_path =  get_combined_img_path(data_source, f'/data/{data_source}', img_path_list)
                question_df.at[global_index, 'combined_img_path'] = combined_img_path
                if not os.path.exists(combined_img_path):
                    combine_image_seq(data_source, f'/data/{data_source}', img_path_list, use_number=True)
                instruction_index = global_index % len(full_instructions)
                question_df.at[global_index, 'task_instruction'] = full_instructions[instruction_index]
                question_df.at[global_index, 'gt_option'] = f'image {event_index + 1}'
                question_df.at[global_index, 'character_info'] = " ".join(list(set(find_characters_in_string(row['all_events'], data_source) )))
                global_index += 1
                
    elif task_name == 'single_grounding_all':
        df['all_events'] = df.filter(regex='^event').apply(lambda row: ''.join(row.values.astype(str)), axis=1)
        column_names = ["data_source", "img_seq_id", "question_id", "task_name", "input_image_type", 
                        "img_path_list", "combined_img_path",
                        "event_to_test","text_to_test", "task_instruction", "gt_option", "character_info"]
        task_instruction_list = task_instructions[task_name[:-4]]
        requirement_template_list = requirement_templates['paired_event_discrimination']
        answer_format_template_list = answer_format_templates[task_name[:-4]]
        full_instructions = generate_combinations(task_instruction_list, requirement_template_list, answer_format_template_list)
        question_df = pd.DataFrame(columns=column_names)
        global_index = 0
        for index, row in df.iterrows():
            event_indices = [int(item.replace('event','')) for item in event_columns]
                
            for event_index in event_indices:
                if pd.isna(row[f"event{event_index}"]):
                    continue
                question_df.at[global_index, 'data_source'] = data_source
                if data_source in ['flintstones', 'pororo']:
                    question_df.at[global_index, 'img_seq_id'] = row['img_seq_id']
                    question_df.at[global_index, 'question_id'] = f'{row['img_seq_id']}_q{global_index}'
                elif data_source in ['vwp']:
                    question_df.at[global_index, 'img_seq_id'] = row['scene_full_id']
                    question_df.at[global_index, 'question_id'] = f'{row['scene_full_id']}_q{global_index}'
                else: #data_source in ['vist']
                        question_df.at[global_index, 'img_seq_id'] = row['album_id']
                        question_df.at[global_index, 'question_id'] = f'{row['story_id']}_q{global_index}'
                        
                question_df.at[global_index, 'task_name'] = task_name
                question_df.at[global_index, 'input_image_type'] = 'combined'
                question_df.at[global_index, 'event_to_test'] = row[f"event{event_index}"]
                question_df.at[global_index, 'text_to_test'] = row[f"text{event_index}"]
                question_df.at[global_index, 'img_to_test'] = row[f"link{event_index}"]
                img_path_list = [row[col] for col in row.index if col.startswith('link')]
                img_path_list = [item for item in img_path_list if not (isinstance(item, float) and math.isnan(item))]
                if not validate_image_paths(img_path_list):
                    continue
                question_df.at[global_index, 'img_path_list'] = img_path_list
                combined_img_path =  get_combined_img_path(data_source, f'/data/{data_source}', img_path_list)
                question_df.at[global_index, 'combined_img_path'] = combined_img_path
                if not os.path.exists(combined_img_path):
                    combine_image_seq(data_source, f'/data/{data_source}', img_path_list, use_number=True)
                instruction_index = global_index % len(full_instructions)
                question_df.at[global_index, 'task_instruction'] = full_instructions[instruction_index]
                question_df.at[global_index, 'gt_option'] = f'image {event_index + 1}'
                question_df.at[global_index, 'character_info'] = " ".join(list(set(find_characters_in_string(row['all_events'], data_source) )))
                global_index += 1
                
                
    elif task_name == 'ordering_images':
        column_names = ["data_source", "img_seq_id", "question_id", "task_name", "input_image_type", 
                        "events_gt", "texts_gt", 
                        "img_path_list_gt", "combined_img_path_gt", 
                        "img_path_list_shuffled", "combined_img_path_shuffled",
                         "options", "gt_answer", "gt_option", 
                        "task_instruction", "answer_format_opg", "answer_format_opt", "character_info"]
        
        task_instruction_list = task_instructions[task_name]
        requirement_template_list = requirement_templates[task_name]
        #answer_format_template_list = answer_format_templates[f'{task_name}_opt']
        full_instructions = full_instructions = [' '.join(combination) for combination in product(task_instruction_list, requirement_template_list)]
     
        question_df = pd.DataFrame(columns=column_names)
        global_index = 0
        for index, row in df.iterrows():
            text_columns = [col for col in row.index if col.startswith('text') and not (isinstance(row[col], float) and math.isnan(row[col]))]
            event_columns = [col for col in row.index if col.startswith('event') and not (isinstance(row[col], float) and math.isnan(row[col]))]
            assert len(text_columns) == len(event_columns)
            link_columns = [col for col in row.index if col.startswith('link') and not (isinstance(row[col], float) and math.isnan(row[col]))]
            all_files_exist = all(os.path.exists(row[col]) for col in link_columns)
            if not all_files_exist:
                continue
            question_df.at[global_index, 'data_source'] = data_source
            if data_source in ['flintstones', 'pororo']:
                question_df.at[global_index, 'img_seq_id'] = row['img_seq_id']
                question_df.at[global_index, 'question_id'] = f'{row['img_seq_id']}_q{global_index}'
            elif data_source in ['vwp']:
                question_df.at[global_index, 'img_seq_id'] = row['scene_full_id']
                question_df.at[global_index, 'question_id'] = f'{row['scene_full_id']}_q{global_index}'
            else: #data_source in ['vist']
                question_df.at[global_index, 'img_seq_id'] = row['album_id']
                question_df.at[global_index, 'question_id'] = f'{row['story_id']}_q{global_index}'
            question_df.at[global_index, 'task_name'] = task_name
            question_df.at[global_index, 'input_image_type'] = 'combined'
            question_df.at[global_index, 'events_gt'] = " ".join([row[event_col] for event_col in event_columns if not (isinstance(row[event_col], float) and math.isnan(row[event_col]))])
            question_df.at[global_index, 'texts_gt'] = " ".join([row[text_col] for text_col in text_columns if not (isinstance(row[text_col], float) and math.isnan(row[text_col]))])
            img_path_list = [row[col] for col in row.index if col.startswith('link')]
            img_path_list = [item for item in img_path_list if not (isinstance(item, float) and math.isnan(item))]
            if not validate_image_paths(img_path_list):
                continue
            question_df.at[global_index, 'img_path_list_gt'] = img_path_list
            combined_img_path_gt = get_combined_img_path(data_source, f'/data/{data_source}', img_path_list, use_number=True)
            question_df.at[global_index, 'combined_img_path_gt'] = combined_img_path_gt
            shuffled_images_for_question, gt_answer = shuffle_list(img_path_list, seed_value=global_index)
            question_df.at[global_index, 'img_path_list_shuffled'] = shuffled_images_for_question
            combined_img_path_shuffled = get_combined_img_path(data_source, f'/data/{data_source}', shuffled_images_for_question, use_number=False)
            question_df.at[global_index, 'combined_img_path_shuffled'] = combined_img_path_shuffled
            if not os.path.exists(combined_img_path_gt):
                combine_image_seq(data_source, f'/data/{data_source}', img_path_list, use_number=True)
            if not os.path.exists(combined_img_path_shuffled):
                combine_image_seq(data_source, f'/data/{data_source}', shuffled_images_for_question, use_number=False)
            question_df.at[global_index, 'gt_answer'] = [f'Image {option}' for option in gt_answer]
            options = create_candidate_options(list(string.ascii_lowercase[:len(text_columns)]), prefix = "Image", gt_answer_list=gt_answer)
            question_df.at[global_index, 'options'], question_df.at[global_index, 'gt_option'] = generate_option_string(options, gt_answer, prefix = "Image", global_index = global_index)
            instruction_index = global_index % len(full_instructions)
            question_df.at[global_index, 'task_instruction'] = full_instructions[instruction_index]
            formats_opt = answer_format_templates[f'{task_name}_opt']
            question_df.at[global_index, 'answer_format_opt'] = formats_opt[global_index % len(formats_opt)]
            formats_opg = answer_format_templates[f'{task_name}_opg']
            question_df.at[global_index, 'answer_format_opg'] = formats_opg[global_index % len(formats_opg)]
            question_df.at[global_index, 'character_info'] = " ".join(list(set(sum([find_characters_in_string(row[f"{text_col}"], data_source) for text_col in text_columns],[]))))
            cot_instruction_index_opg = global_index % len(orderimg_instruction_opg_cot)
            question_df.at[global_index, 'task_instruction_cot_opg'] = orderimg_instruction_opg_cot[cot_instruction_index_opg]
            cot_instruction_index_opt = global_index % len(orderimg_instruction_opt_cot)
            question_df.at[global_index, 'task_instruction_cot_opt'] = orderimg_instruction_opt_cot[cot_instruction_index_opt]
            
            global_index += 1
            
        
    elif task_name == 'ordering_texts':
        column_names = ["data_source", "img_seq_id", "question_id", "task_name", "input_image_type", 
                        "img_path_list_gt", "combined_img_path_gt", 
                        "events_gt", "texts_gt",
                        "events_shuffled", "texts_shuffled", 
                        "gt_answer_text", "gt_answer", "gt_option", "options",
                        "task_instruction",  "character_info",]
        task_instruction_list = task_instructions[task_name]
        requirement_template_list = requirement_templates[task_name]
        full_instructions = [' '.join(combination) for combination in product(task_instruction_list, requirement_template_list)]
     
        question_df = pd.DataFrame(columns=column_names)
        global_index = 0
        for index, row in df.iterrows():
            text_columns = [col for col in row.index if col.startswith('text') and not (isinstance(row[col], float) and math.isnan(row[col]))]
            event_columns = [col for col in row.index if col.startswith('event') and not (isinstance(row[col], float) and math.isnan(row[col]))]
            assert len(text_columns) == len(event_columns)
            question_df.at[global_index, 'data_source'] = data_source
            if data_source in ['flintstones', 'pororo']:
                question_df.at[global_index, 'img_seq_id'] = row['img_seq_id']
                question_df.at[global_index, 'question_id'] = f'{row['img_seq_id']}_q{global_index}'
            elif data_source in ['vwp']:
                question_df.at[global_index, 'img_seq_id'] = row['scene_full_id']
                question_df.at[global_index, 'question_id'] = f'{row['scene_full_id']}_q{global_index}'
            else: #data_source in ['vist']
                question_df.at[global_index, 'img_seq_id'] = row['album_id']
                question_df.at[global_index, 'question_id'] = f'{row['story_id']}_q{global_index}'
            question_df.at[global_index, 'task_name'] = task_name
            question_df.at[global_index, 'input_image_type'] = 'combined'
            img_path_list = [row[col] for col in row.index if col.startswith('link')]
            img_path_list = [item for item in img_path_list if not (isinstance(item, float) and math.isnan(item))]
            if not validate_image_paths(img_path_list):
                continue
            question_df.at[global_index, 'img_path_list_gt'] = img_path_list
            combined_img_path_gt = get_combined_img_path(data_source, f'/data/{data_source}', img_path_list, use_number=True)
            question_df.at[global_index, 'combined_img_path_gt'] = combined_img_path_gt
            if not os.path.exists(combined_img_path_gt):
                combine_image_seq(data_source, f'/data/{data_source}', img_path_list, use_number=True)
            events_list_gt = [row[event_col] for event_col in event_columns]
            texts_list_gt = [row[text_col] for text_col in text_columns]
            question_df.at[global_index, 'events_gt'] = " ".join(events_list_gt)
            question_df.at[global_index, 'texts_gt'] = " ".join(texts_list_gt)
            shuffled_texts_for_question, texts_gt_answer = shuffle_list(texts_list_gt, seed_value=global_index)
            shuffled_events_for_question, events_gt_answer = shuffle_list(events_list_gt, seed_value=global_index)
            assert texts_gt_answer == events_gt_answer
            question_df.at[global_index, 'events_shuffled'] = shuffled_events_for_question
            question_df.at[global_index, 'texts_shuffled'] =shuffled_texts_for_question
            question_df.at[global_index, 'gt_answer_text'] = [f'Sentence {option}' for option in texts_gt_answer]
            question_df.at[global_index, 'gt_answer'] = [f'Sentence {option}' for option in texts_gt_answer]
            options = create_candidate_options(list(string.ascii_lowercase[:len(text_columns)]), prefix = "Sentence", gt_answer_list=texts_gt_answer)
            question_df.at[global_index, 'options'], question_df.at[global_index, 'gt_option'] = generate_option_string(options, texts_gt_answer, prefix = "Sentence", global_index = global_index)
            instruction_index = global_index % len(full_instructions)
            question_df.at[global_index, 'task_instruction'] = full_instructions[instruction_index]
            formats_opt = answer_format_templates[f'{task_name}_opt']
            question_df.at[global_index, 'answer_format_opt'] = formats_opt[global_index % len(formats_opt)]
            formats_opg = answer_format_templates[f'{task_name}_opg']
            question_df.at[global_index, 'answer_format_opg'] = formats_opg[global_index % len(formats_opg)]
            question_df.at[global_index, 'character_info'] = " ".join(list(set(sum([find_characters_in_string(row[f"{text_col}"], data_source) for text_col in text_columns],[]))))
            blind_index_opg = global_index % len(blind_instruction_ordersent_opg)
            question_df.at[global_index, 'blind_instruction_opg'] = blind_instruction_ordersent_opg[blind_index_opg]
            # question_df.at[global_index, 'blind_instruction_opg_reason'] = blind_instruction_ordersent_opg_reason
            blind_index_opt = global_index % len(blind_instruction_ordersent_opt)
            question_df.at[global_index, 'blind_instruction_opt'] = blind_instruction_ordersent_opt[blind_index_opt]
            # question_df.at[global_index, 'blind_instruction_opt_reason'] = blind_instruction_ordersent_opt_reason
            cot_instruction_index_opg = global_index % len(ordersent_instruction_opg_cot)
            question_df.at[global_index, 'task_instruction_cot_opg'] = ordersent_instruction_opg_cot[cot_instruction_index_opg]
            cot_instruction_index_opt = global_index % len(ordersent_instruction_opt_cot)
            question_df.at[global_index, 'task_instruction_cot_opt'] = ordersent_instruction_opt_cot[cot_instruction_index_opt]
            global_index += 1
            
    elif task_name == 'reversed_ordering_texts':
        column_names = ["data_source", "img_seq_id", "question_id", "task_name", "input_image_type", 
                        "img_path_list_reversed", "combined_img_path_reversed",
                        "events_gt", "texts_gt",
                        "events_shuffled", "texts_shuffled", 
                        "gt_answer_text", "gt_answer", "gt_option", "options",
                        "task_instruction",  "character_info",]
        task_instruction_list = task_instructions[task_name]
        requirement_template_list = requirement_templates[task_name]
        full_instructions = [' '.join(combination) for combination in product(task_instruction_list, requirement_template_list)]
     
        question_df = pd.DataFrame(columns=column_names)
        global_index = 0
        for index, row in df.iterrows():
            text_columns = [col for col in row.index if col.startswith('text') and not (isinstance(row[col], float) and math.isnan(row[col]))]
            event_columns = [col for col in row.index if col.startswith('event') and not (isinstance(row[col], float) and math.isnan(row[col]))]
            assert len(text_columns) == len(event_columns)
            question_df.at[global_index, 'data_source'] = data_source
            if data_source in ['flintstones', 'pororo']:
                question_df.at[global_index, 'img_seq_id'] = row['img_seq_id']
                question_df.at[global_index, 'question_id'] = f'{row['img_seq_id']}_q{global_index}'
            elif data_source in ['vwp']:
                question_df.at[global_index, 'img_seq_id'] = row['scene_full_id']
                question_df.at[global_index, 'question_id'] = f'{row['scene_full_id']}_q{global_index}'
            else: #data_source in ['vist']
                    question_df.at[global_index, 'img_seq_id'] = row['album_id']
                    question_df.at[global_index, 'question_id'] = f'{row['story_id']}_q{global_index}'
            question_df.at[global_index, 'task_name'] = task_name
            question_df.at[global_index, 'input_image_type'] = 'combined'
            img_path_list = [row[col] for col in row.index if col.startswith('link')]
            img_path_list = [item for item in img_path_list if not (isinstance(item, float) and math.isnan(item))]
            if not validate_image_paths(img_path_list):
                continue
            question_df.at[global_index, 'img_path_list_reversed'] = img_path_list[::-1]                
            combined_img_path_reversed = get_combined_img_path(data_source, f'/data/{data_source}', img_path_list[::-1], use_number=True, reversed=True)
            question_df.at[global_index, 'combined_img_path_reversed'] = combined_img_path_reversed
            if not os.path.exists(combined_img_path_reversed):
                combine_image_seq(data_source, f'/data/{data_source}', img_path_list[::-1], use_number=True, reversed=True)
            
            
            events_list_gt = [row[event_col] for event_col in event_columns]
            texts_list_gt = [row[text_col] for text_col in text_columns]
            question_df.at[global_index, 'events_gt'] = " ".join(events_list_gt)
            question_df.at[global_index, 'texts_gt'] = " ".join(texts_list_gt)
            shuffled_texts_for_question, texts_gt_answer = shuffle_list(texts_list_gt, seed_value=global_index)
            shuffled_events_for_question, events_gt_answer = shuffle_list(events_list_gt, seed_value=global_index)
            assert texts_gt_answer == events_gt_answer
            question_df.at[global_index, 'events_shuffled'] = shuffled_events_for_question
            question_df.at[global_index, 'texts_shuffled'] =shuffled_texts_for_question
            question_df.at[global_index, 'gt_answer_text'] = [f'Sentence {option}' for option in texts_gt_answer]
            question_df.at[global_index, 'gt_answer'] = [f'Sentence {option}' for option in texts_gt_answer]
            options = create_candidate_options(list(string.ascii_lowercase[:len(text_columns)]), prefix = "Sentence", gt_answer_list=texts_gt_answer)
            question_df.at[global_index, 'options'], question_df.at[global_index, 'gt_option'] = generate_option_string(options, texts_gt_answer, prefix = "Sentence", global_index = global_index)
            instruction_index = global_index % len(full_instructions)
            question_df.at[global_index, 'task_instruction'] = full_instructions[instruction_index]
            formats_opt = answer_format_templates[f'{task_name}_opt']
            question_df.at[global_index, 'answer_format_opt'] = formats_opt[global_index % len(formats_opt)]
            formats_opg = answer_format_templates[f'{task_name}_opg']
            question_df.at[global_index, 'answer_format_opg'] = formats_opg[global_index % len(formats_opg)]
            question_df.at[global_index, 'character_info'] = " ".join(list(set(sum([find_characters_in_string(row[f"{text_col}"], data_source) for text_col in text_columns],[]))))
            blind_index_opg = global_index % len(blind_instruction_ordersent_opg)
            question_df.at[global_index, 'blind_instruction_opg'] = blind_instruction_ordersent_opg[blind_index_opg]
            # question_df.at[global_index, 'blind_instruction_opg_reason'] = blind_instruction_ordersent_opg_reason
            blind_index_opt = global_index % len(blind_instruction_ordersent_opt)
            question_df.at[global_index, 'blind_instruction_opt'] = blind_instruction_ordersent_opt[blind_index_opt]
            # question_df.at[global_index, 'blind_instruction_opt_reason'] = blind_instruction_ordersent_opt_reason
            cot_instruction_index_opg = global_index % len(ordersent_instruction_opg_cot)
            question_df.at[global_index, 'task_instruction_cot_opg'] = ordersent_instruction_opg_cot[cot_instruction_index_opg]
            cot_instruction_index_opt = global_index % len(ordersent_instruction_opt_cot)
            question_df.at[global_index, 'task_instruction_cot_opt'] = ordersent_instruction_opt_cot[cot_instruction_index_opt]
            global_index += 1
            
    
    
    else:
        print("Invalid task name")
        sys.exit(0)

        
    return question_df


def main():
    parser = argparse.ArgumentParser(description="Generate prompts for benchmarking")
    parser.add_argument("--data_source", type=str, default="vist", help="Data source")
    parser.add_argument("--save_path", type=str, default="/data/benchmark")
    args = parser.parse_args()
    
    data_source = args.data_source
    
    
    events_path = f'/data/{data_source}/filtered_{data_source}_events.pkl'
    df_events = pd.read_pickle(events_path)
    df_events = df_events.replace(
        to_replace=r'/scratch/song0018/temporal_data/', 
        value='/data/', 
        regex=True
    )
    if data_source != 'vwp':
        df_events = df_events.dropna()
        
        
    triples_path = f'/data/{data_source}/filtered_{data_source}_triples.pkl'
    df_triples = pd.read_pickle(triples_path)
    df_triples = df_triples.replace(
        to_replace=r'/scratch/song0018/temporal_data/', 
        value='/data/', 
        regex=True
    )
    if data_source != 'vwp':
        df_triples = df_triples.dropna()
        
    
        
    
    single_grounding_df = create_questions(df_events, data_source, 'single_grounding_paired')
    single_grounding_df.to_pickle(os.path.join(args.save_path, f"{data_source}_single_grounding_paired_test.pkl"))
    
    single_grounding_triple_df = create_questions(df_triples, data_source, 'single_grounding_triple')
    single_grounding_triple_df.to_pickle(os.path.join(args.save_path, f"{data_source}_single_grounding_triple_test.pkl"))
    
    single_grounding_all_story_df = create_questions(df_events, data_source, 'single_grounding_all_story')
    single_grounding_all_story_df.to_pickle(os.path.join(args.save_path, f"{data_source}_single_grounding_all_story_test.pkl"))
      
    single_grounding_all_df = create_questions(df_events, data_source, 'single_grounding_all')
    single_grounding_all_df.to_pickle(os.path.join(args.save_path, f"{data_source}_single_grounding_all_test.pkl"))
   
    paired_grounding_one_image_df = create_questions(df_events, data_source, task_name="paired_grounding_one_image")
    paired_grounding_one_image_df.to_pickle(os.path.join(args.save_path, f"{data_source}_paired_grounding_one_image_test.pkl"))
   
    paired_grounding_one_text_df = create_questions(df_events, data_source, task_name="paired_grounding_one_text")
    paired_grounding_one_text_df.to_pickle(os.path.join(args.save_path, f"{data_source}_paired_grounding_one_text_test.pkl"))
   
    paired_event_discrimination_df = create_questions(df_events, data_source, task_name="paired_event_discrimination")
    paired_event_discrimination_df.to_pickle(os.path.join(args.save_path, f"{data_source}_paired_event_discrimination_test.pkl"))
    
    paired_grounding_one_image_df = create_questions(df_triples, data_source, task_name="triple_grounding_one_image")
    paired_grounding_one_image_df.to_pickle(os.path.join(args.save_path, f"{data_source}_triple_grounding_one_image_test.pkl"))
   
    paired_grounding_one_text_df = create_questions(df_triples, data_source, task_name="triple_grounding_one_text")
    paired_grounding_one_text_df.to_pickle(os.path.join(args.save_path, f"{data_source}_triple_grounding_one_text_test.pkl"))
   
    paired_event_discrimination_df = create_questions(df_triples, data_source, task_name="triple_event_discrimination")
    paired_event_discrimination_df.to_pickle(os.path.join(args.save_path, f"{data_source}_triple_event_discrimination_test.pkl"))
   
    ordering_images_df = create_questions(df_events, data_source, task_name="ordering_images")
    ordering_images_df.to_pickle(os.path.join(args.save_path, f"{data_source}_ordering_images_test.pkl"))
   
    ordering_texts_df = create_questions(df_events, data_source, task_name="ordering_texts")
    ordering_texts_df.to_pickle(os.path.join(args.save_path, f"{data_source}_ordering_texts_test.pkl"))
    
    
    
    print(f"Prompts generated successfully!-{data_source}")

    
    
if __name__ == "__main__":
    main()
        
