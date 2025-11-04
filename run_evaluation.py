import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import sys
import os
import re
import warnings
warnings.filterwarnings("ignore")

data_sources = [
    'flintstones', 
    'pororo', 
    'vwp', 
    'vist']

models = [
    'deepseek-vl2-tiny','deepseek-vl2-small',
    'InternVL2_5-1B', 'InternVL2_5-1B-MPO', 
    'InternVL2_5-8B', 'InternVL2_5-8B-MPO',
    'InternVL2_5-26B','InternVL2_5-26B-MPO', 
    'InternVL2_5-78B', 'InternVL2_5-78B-MPO', 
    'Janus-Pro-1B', 'Janus-Pro-7B',
    'llava-interleave-qwen-0.5b-hf', 'llava-interleave-qwen-7b-hf', 'llava-interleave-qwen-7b-dpo-hf',
    'llava-onevision-qwen2-0.5b-ov', 'llava-onevision-qwen2-0.5b-si',
    'llava-onevision-qwen2-7b-ov', 'llava-onevision-qwen2-7b-si',
    'llava-onevision-qwen2-72b-ov-sft', 'llava-onevision-qwen2-72b-si',
    'LLaVA-NeXT-Video-7B-hf', 'LLaVA-NeXT-Video-7B-DPO-hf', 
    'LLaVA-NeXT-Video-34B-hf', 'LLaVA-NeXT-Video-34B-DPO-hf',
    'LongVA-7B', 'LongVA-7B-DPO',
    'Mantis-8B-Idefics2', 'Mantis-8B-siglip-llama3',
    'Phi-3-vision-128k-instruct', 'Phi-3.5-vision-instruct',
    'Qwen2-VL-2B', 'Qwen2-VL-2B-Instruct', 
    'Qwen2-VL-7B', 'Qwen2-VL-7B-Instruct', 
    'Qwen2-VL-72B','Qwen2-VL-72B-Instruct', 
    'GPT4o',
    #'Llama-3.1-8B', 'Phi-3.5-mini-instruct', 'Qwen2.5-72B-Instruct'
    ]

tasks = [ 
        "paired_event_discrimination", "triple_event_discrimination",
        "ordering_texts_opt_event", "ordering_texts_opt_story",
        "ordering_images_opt_event", "ordering_images_opt_story",   
         ]

models_llm = ['Llama-3.1-8B', 'Phi-3.5-mini-instruct', 'Qwen2.5-72B-Instruct']


def extract_after_assistant(text):
    return re.split(r'(?i)assistant', text, 1)[-1]

def extract_after_answer(text):
    return re.split(r'(?i)The answer is', text, 1)[-1]

def get_answer_from_choice(choice, string):
    # Split the string into components based on semicolon
    options = string.split("; ")
    
    # Create a dictionary to map letters (A, B, C) to their corresponding values
    answer_map = {}
    for option in options:
        # Split each option into letter and its corresponding value
        letter, value = option.split(". ")
        answer_map[letter.strip()] = value.strip()
    
    # Return the value corresponding to the input choice
    return answer_map.get(choice.upper(), "Invalid choice")

strings = [
    'A. True; B. Unsure; C. False.',
    'A. False; B. Unsure; C. True.',
    'A. False; B. True; C. Unsure.',
    'A. Unsure; B. True; C. False.'
]

print(get_answer_from_choice('A', strings[0]))
print(get_answer_from_choice('A', strings[1]))
print(get_answer_from_choice('A', strings[2]))
print(get_answer_from_choice('A', strings[3]))

def extract_AB_letter(text):
    pattern1 = r'Option\s+([AB])'  # Matches "Option A" or "Option B"
    pattern2 = r'\b([AB])\.\s*(?:True|False)'  # Matches "A. True" or "B. False"
    pattern3 = r'\b([AB])\s*$'  # Matches standalone "A" or "B"
    pattern4 = r'Option\s*\[([AB])\]'  # Matches "Option [A]" or "Option [B]"
    
    match1 = re.search(pattern1, text)
    match2 = re.search(pattern2, text)
    match3 = re.search(pattern3, text)
    match4 = re.search(pattern4, text)  # Add new pattern match
    
    if match1:
        return match1.group(1)
    elif match2:
        return match2.group(1)
    elif match3:
        return match3.group(1)
    elif match4:  # Add new condition
        return match4.group(1)
    else:
        return 'UNK'
    
def extract_after_assistant(text):
    return re.split(r'(?i)assistant', text, 1)[-1]
        
def process_predictions(model, data_source, task, pred_folder):
    ### Load the predictions
    try:
        larger_models = ['72b', '78b', '110b']
        if any(size in model.lower() for size in larger_models):
            load_8bit = True
            suffix = '8bit'
        else:
            load_8bit = False
            suffix = 'full'
            
        
        df = pd.read_pickle(f'temporal_data/{pred_folder}/{model}/{data_source}_{task}_preds_{suffix}.pkl')
        event_columns = [col for col in df.columns if col.startswith('event')]
        if event_columns:
            # List of strings to check for
            invalid_events = ['NO_EVENT', 'No_Event', 'no_event', "NO_Event", 
                            "good time", "great time", "wonderful time", 
                            "have fun", "had fun"]
            # invalid_events = ['NO_EVENT', 'No_Event', 'no_event', "NO_Event"]
            for col in event_columns:
                df[col] = df[col].astype(str)
            mask = ~df[event_columns].apply(lambda x: x.str.contains('|'.join(invalid_events), case=False, na=False)).any(axis=1)
            df = df[mask]
    
            
    except FileNotFoundError:
        print(f'File not found: {model}/{data_source}_{task}_preds_{suffix}.pkl')
        return None
    
    
    
    if 'deepseek' in model or 'Janus' in model:
        df['preds'] = df['preds'].str.replace(' <｜end▁of▁sentence｜>', '')
        if 'grounding' in task:
            df['cleaned_preds'] = df['preds'].str.extract(r'(\d+)').fillna('UNK')
        elif 'discrimination' in task:
            df['cleaned_preds'] = df['preds'].str.extract(r'\b([AB])\b').fillna('UNK')
        elif 'ordering' in task:
            df['cleaned_preds'] = df['preds'].str.extract(r'\b([ABCDE])\b').fillna('UNK')
        else:
            print(f'Unknown task: {task}')
            
    elif model == 'Llama-3.1-8B':
        df['cleaned_preds'] = df['preds'].apply(extract_after_answer)
        if 'discrimination' in task:
            df['cleaned_preds'] = df['cleaned_preds'].str.extract(r'\b([AB])\b').fillna('UNK')
        elif 'ordering' in task:
            df['cleaned_preds'] = df['cleaned_preds'].str.extract(r'\b([ABCDE])\b').fillna('UNK')
            
    elif model == 'Phi-3.5-mini-instruct':
        if 'discrimination' in task:
            df['cleaned_preds'] = df['preds'].str.extract(r'\b([AB])\b').fillna('UNK')
        elif 'ordering' in task:
            df['cleaned_preds'] = df['preds'].str.extract(r'\b([ABCDE])\b').fillna('UNK')
        

    elif model == 'Qwen2.5-72B-Instruct':
        if 'discrimination' in task:
            df['cleaned_preds'] = df['preds'].str.extract(r'\b([AB])\b').fillna('UNK')
        elif 'ordering' in task:
            df['cleaned_preds'] = df['preds'].str.extract(r'\b([ABCDE])\b').fillna('UNK')
       
        
    
    elif 'InternVL' in model:
        if 'grounding' in task:
            df['cleaned_preds'] = df['preds'].str.extract(r'(\d+)').fillna('UNK')
        elif 'discrimination' in task:
            df['cleaned_preds'] = df['preds'].apply(extract_AB_letter)
        elif 'ordering' in task:
            df['cleaned_preds'] = df['preds'].str.extract(r'\b([ABCDE])\b').fillna('UNK')
        else:
            print(f'Unknown task: {task}')
            
    elif 'interleave' in model or 'LLaVA-NeXT-Video' in model:
        df['cleaned_preds'] = df['preds'].apply(extract_after_assistant)
        if 'grounding' in task:
            df['cleaned_preds'] = df['cleaned_preds'].str.extract(r'(\d+)').fillna('UNK')
        elif 'discrimination' in task:
            df['cleaned_preds'] = df['cleaned_preds'].str.extract(r'\b([AB])\b').fillna('UNK')
        elif 'ordering' in task:
            df['cleaned_preds'] = df['cleaned_preds'].str.extract(r'\b([ABCDE])\b').fillna('UNK')
    
    elif 'onevision' in model or 'LongVA' in model or 'Mantis' in model or 'Phi' in model or 'Qwen2' in model:
        if 'grounding' in task:
            df['cleaned_preds'] = df['preds'].str.extract(r'(\d+)').fillna('UNK')
        elif 'discrimination' in task:
            df['cleaned_preds'] = df['preds'].str.extract(r'\b([AB])\b').fillna('UNK')
        elif 'ordering' in task:
            df['cleaned_preds'] = df['preds'].str.extract(r'\b([ABCDE])\b').fillna('UNK')
            
    elif model == 'GPT4o':
        if 'grounding' in task:
            df['cleaned_preds'] = df['preds'].str.extract(r'(\d+)').fillna('UNK')
        elif 'discrimination' in task:
            df['cleaned_preds'] = df['preds'].str.extract(r'\b([AB])\b').fillna('UNK')
        elif 'ordering' in task:
            df['cleaned_preds'] = df['preds'].str.extract(r'\b([ABCDE])\b').fillna('UNK')
        
        
            
    else:
        print(f'Unknown model: {model}')
        
    df['pass'] = ""    
    for ind, row in df.iterrows():
        if row['cleaned_preds'] == row['gt_option'][-1]:
            df.at[ind, 'pass'] = 'true'
        else:
            df.at[ind, 'pass'] = 'false'
        
    return df
    
        
def analyze_grounding_vs_main_task(model, data_source, main_task, pred_folder='predictions/combined'):
    
    if 'story' in main_task and 'ordering' in main_task:
        grounding_task = 'single_grounding_all_story'
    elif 'event' in main_task and 'ordering' in main_task:
        grounding_task = 'single_grounding_all'
    elif 'paired_event' in main_task:
        grounding_task = 'single_grounding_paired'
    elif 'triple_event' in main_task:
        grounding_task = 'single_grounding_triple'
    else:
        print(f'Unknown grounding task: {main_task}')
        return None

    df_grounding = process_predictions(model, data_source, grounding_task, pred_folder)
    df_main = process_predictions(model, data_source, main_task, pred_folder)
    print(df_main.shape)
    if df_grounding is None or df_main is None:
        return None
    
    if 'images' not in main_task: 
        df_llm1 = process_predictions(models_llm[0], data_source, main_task, 'predictions/blind')
        df_llm2 = process_predictions(models_llm[1], data_source, main_task, 'predictions/blind')
        df_llm3 = process_predictions(models_llm[2], data_source, main_task, 'predictions/blind')
        
        assert len(df_llm1) == len(df_llm2) == len(df_llm3) == len(df_main)
        
        indices_to_drop = []
        
        for ind, row in df_main.iterrows():
            passes = [df_llm1.at[ind, 'pass'], df_llm2.at[ind, 'pass'], df_llm3.at[ind, 'pass']]
            true_count = passes.count('true')
            if true_count >= 2:
                indices_to_drop.append(ind)
     
        df_main = df_main.drop(indices_to_drop)
        
        print("Dataframe after dropping language-only results")    
        print(df_main.shape)
                
    
    
    
    
    
        
    # Initialize gt_passes column
    df_main['gt_passes'] = ""
    
    
    
    
    # Process each row
    for ind, row in df_main.iterrows():
        img_seq_id = row['img_seq_id']
        if 'ordering' in main_task:
            matching_rows = df_grounding[df_grounding['img_seq_id'] == img_seq_id]
            if matching_rows.empty:
                print(f'No matching rows for img_seq_id: {img_seq_id}')
                continue
            gt_passes = list(matching_rows['pass'])

            
        else:
            if 'paired' in main_task:
                event_indices = eval(row['event_pair_tuple'])
            elif 'triple' in main_task:
                event_indices = eval(row['event_triple_tuple'])
            gt_passes = []
            for event_index in event_indices:
                matching_row = df_grounding[
                    (df_grounding['img_seq_id'] == img_seq_id) & 
                    (df_grounding['gt_option'].str.endswith(str(event_index+1)))
                ]
                if matching_row.empty:
                    print(f'No matching row for img_seq_id: {img_seq_id}, event_index: {event_index}')
                    continue
                else:
                    gt_passes.append(matching_row.iloc[0]['pass'])
           
        df_main.at[ind, 'gt_passes'] = gt_passes
    print(f"Finished processing {model} - {data_source} - {main_task}")
   
    
    # Calculate metrics
    total_cases = len(df_main['gt_passes'])
    count_all_true_grounding = sum(1 for passes in df_main['gt_passes'] if not 'false' in passes)
    count_true_main = sum(1 for pass_val in df_main['pass'] if pass_val == 'true')
    count_both_true = sum(1 for passes, pass_val in zip(df_main['gt_passes'], df_main['pass']) 
                         if not 'false' in passes and pass_val == 'true')
    
    # Create masks for conditional analysis
    grounding_pass_mask = [not 'false' in passes for passes in df_main['gt_passes']]
    main_pass_mask = [pass_val == 'true' for pass_val in df_main['pass']]
    
    # Calculate statistics for cases that pass grounding
    grounding_pass_count = sum(grounding_pass_mask)
    main_pass_given_grounding = sum(1 for g, m in zip(grounding_pass_mask, main_pass_mask) if g and m)
    
    # Calculate statistics for cases that fail grounding
    grounding_fail_count = sum(not x for x in grounding_pass_mask)
    main_pass_given_no_grounding = sum(1 for g, m in zip(grounding_pass_mask, main_pass_mask) if not g and m)
    
    # Compile results
    results = {
        'total_cases': total_cases,
        'all_grounding_success': {
            'count': count_all_true_grounding,
            'percentage': (count_all_true_grounding / total_cases ) if total_cases > 0 else 0
        },
        'main_task_success': {
            'count': count_true_main,
            'percentage': (count_true_main / total_cases ) if total_cases > 0 else 0
        },
        'both_success': {
            'count': count_both_true,
            'percentage': (count_both_true / total_cases ) if total_cases > 0 else 0
        },
        'conditional_on_grounding_pass': {
            'count': main_pass_given_grounding,
            'total': grounding_pass_count,
            'percentage': (main_pass_given_grounding / grounding_pass_count ) if grounding_pass_count > 0 else 0
        },
        'conditional_on_grounding_fail': {
            'count': main_pass_given_no_grounding,
            'total': grounding_fail_count,
            'percentage': (main_pass_given_no_grounding / grounding_fail_count ) if grounding_fail_count > 0 else 0
        }
    }
    
    if 'ordering' in main_task:
        for pass_count in [1, 2 , 3, 4]:
            pass_count_mask = [passes.count('true') >= pass_count for passes in df_main['gt_passes']]
            pass_count_cases = sum(pass_count_mask)
            
            if pass_count_cases > 0:
                main_success = sum(1 for mask, main_pass in zip(pass_count_mask, df_main['pass']) 
                                if mask and main_pass == 'true')
                
                results[f'grounding_{pass_count}_passes'] = {
                    'count': pass_count_cases,
                    'main_task_success': {
                        'count': main_success,
                        'percentage': (main_success / pass_count_cases )
                    }
                }
            else:
                results[f'grounding_{pass_count}_passes'] = {
                    'count': 0,
                    'main_task_success': {
                        'count': 0,
                        'percentage': 0
                    }
                }
    elif 'paired' in main_task:
        for pass_count in [1]:
            pass_count_mask = [passes.count('true') >= pass_count for passes in df_main['gt_passes']]
            pass_count_cases = sum(pass_count_mask)
            
            if pass_count_cases > 0:
                main_success = sum(1 for mask, main_pass in zip(pass_count_mask, df_main['pass']) 
                                if mask and main_pass == 'true')
                
                results[f'grounding_{pass_count}_passes'] = {
                    'count': pass_count_cases,
                    'main_task_success': {
                        'count': main_success,
                        'percentage': (main_success / pass_count_cases )
                    }
                }
    elif 'triple' in main_task:
        for pass_count in [1, 2]:
            pass_count_mask = [passes.count('true') >= pass_count for passes in df_main['gt_passes']]
            pass_count_cases = sum(pass_count_mask)
            
            if pass_count_cases > 0:
                main_success = sum(1 for mask, main_pass in zip(pass_count_mask, df_main['pass']) 
                                if mask and main_pass == 'true')
                
                results[f'grounding_{pass_count}_passes'] = {
                    'count': pass_count_cases,
                    'main_task_success': {
                        'count': main_success,
                        'percentage': (main_success / pass_count_cases )
                    }
                }
    
    
    return results
    
def analyze_task_performance(task, models_to_analyze=None, pred_folder='predictions/combined'):
    # Initialize base columns
    base_columns = [
        'total_cases',
        'all_grounding_success',
        'main_task_success', 
        'all_grounding_and_main_success',
        'grounding_acc',
        'main_task_acc',
        'both_tasks_acc',
        'main_acc_given_grounding_pass',
        'main_acc_given_grounding_fail',
    ]
    
    # Determine pass count columns based on task type
    pass_count_columns = []
    if 'ordering' in task:
        pass_count_range = range(1, 5)  # 1,2,3,4
    elif 'triple' in task:
        pass_count_range = range(1, 3)  # 1,2
    elif 'paired' in task:
        pass_count_range = range(1, 2)  # 1
    else:
        pass_count_range = []
        
    # Add columns for each pass count
    for i in pass_count_range:
        pass_count_columns.extend([
            f'{i}_pass_count',
            f'main_acc_given_{i}_pass'
        ])
    
    # Initialize DataFrame with all columns
    results_df = pd.DataFrame(columns=base_columns + pass_count_columns)
    
    # Use provided models list or default to all models
    models_to_analyze = models_to_analyze or models
    
    # Analyze each model
    for model in models_to_analyze:
        # Initialize base counters
        total_cases = 0
        total_grounding_success = 0
        total_main_success = 0
        total_both_success = 0
        total_grounding_pass = 0
        total_main_given_grounding = 0
        total_grounding_fail = 0
        total_main_given_fail = 0
        
        # Initialize pass count trackers based on task type
        pass_counts = {i: 0 for i in pass_count_range}
        main_success_by_passes = {i: 0 for i in pass_count_range}
        
        # Analyze each data source
        for data_source in data_sources:
            results = analyze_grounding_vs_main_task(
                model=model,
                data_source=data_source,
                main_task=task,
                pred_folder=pred_folder
            )

            if results is not None:
                # Accumulate base counts
                total_cases += results['total_cases']
                total_grounding_success += results['all_grounding_success']['count']
                total_main_success += results['main_task_success']['count']
                total_both_success += results['both_success']['count']
                total_grounding_pass += results['conditional_on_grounding_pass']['total']
                total_main_given_grounding += results['conditional_on_grounding_pass']['count']
                total_grounding_fail += results['conditional_on_grounding_fail']['total']
                total_main_given_fail += results['conditional_on_grounding_fail']['count']
                
                # Accumulate pass counts based on task type
                for pass_count in pass_count_range:
                    if f'grounding_{pass_count}_passes' in results:
                        pass_counts[pass_count] += results[f'grounding_{pass_count}_passes']['count']
                        main_success_by_passes[pass_count] += results[f'grounding_{pass_count}_passes']['main_task_success']['count']

        # Add results to DataFrame if we have data
        if total_cases > 0:
            # Initialize with base metrics
            model_results = {
                'total_cases': total_cases,
                'all_grounding_success': total_grounding_success,
                'main_task_success': total_main_success,
                'all_grounding_and_main_success': total_both_success,
                'grounding_acc': (total_grounding_success/total_cases),
                'main_task_acc': (total_main_success/total_cases),
                # 'both_tasks_acc': (total_both_success/total_cases),
                'main_acc_given_grounding_pass': (total_main_given_grounding/total_grounding_pass) if total_grounding_pass > 0 else None,
                'main_acc_given_grounding_fail': (total_main_given_fail/total_grounding_fail) if total_grounding_fail > 0 else None,
            }
            
            # Add pass count metrics
            for pass_count in pass_count_range:
                model_results[f'{pass_count}_pass_count'] = pass_counts[pass_count]
                model_results[f'main_acc_given_{pass_count}_pass'] = (
                    main_success_by_passes[pass_count]/pass_counts[pass_count]
                ) if pass_counts[pass_count] > 0 else None
            
            results_df.loc[model] = model_results

    # Round results to 4 decimal places
    results_df = results_df.round(4)
    
    return results_df

# Example usage:
for task in tasks:
    print(f"\nProcessing task: {task}")
    output_dir = 'temporal_data/evaluation/new_test'
    os.makedirs(output_dir, exist_ok=True)
    
    results = analyze_task_performance(task)
    
    output_file = f"{output_dir}/{task}_results.csv"
    results.to_csv(output_file)
    print(f"Saved results to {output_file}")
    
    

    
    
    
    
    
