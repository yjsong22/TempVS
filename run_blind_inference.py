from matplotlib import path, pyplot as plt
import pandas as pd
import argparse
from tqdm import tqdm
import sys
import torch
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import BitsAndBytesConfig


import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference using LLMs')
    parser.add_argument('--data_source', type=str, default='flintstones',
                        help='Source of the data (e.g., flintstones, pororo, vist, vwp)')
    parser.add_argument('--task', type=str, default='paired_event_discrimination',
                        choices=[
                                 'paired_event_discrimination',  
                                 'triple_event_discrimination', 
                                 'ordering_texts_opg_story', 'ordering_texts_opg_event',
                                 'ordering_texts_opt_story', 'ordering_texts_opt_event'
                                 ],
                        help='Type of task to perform')
    parser.add_argument('--model_id', type=str, 
                        default= 'microsoft/Phi-3.5-mini-instruct', 
                        choices= ['microsoft/Phi-3.5-mini-instruct', 'Qwen/Qwen2.5-72B-Instruct', 'meta-llama/Llama-3.1-8B'],
                        help='Pretrained model path')

    return parser.parse_args()


def process_benchmark(row, task):
    answer_str = '\nThe answer is: '

    
    if 'event_discrimination' in task:
        option_text = '\nOptions are: '+row['option_text']
        task_inst = '\n'+row['blind_instruction']
        statement = '\nThe statement is: '+row['statement_text']
        question_text = statement + task_inst + option_text + answer_str        
    
    elif 'ordering_texts' in task:
        shuffled_story = '\nThe shuffled sentences are:\n' + '\n'.join(f"Sentence {chr(97 + i)}: {text}" for i, text in enumerate(row['texts_shuffled'])) 
        shuffled_event = '\nThe shuffled events are:\n' + '\n'.join(f"Sentence {chr(97 + i)}: {event}" for i, event in enumerate(row['events_shuffled']))  
        
        if 'story' in task:
            shuffled_text = shuffled_story
        else:
            shuffled_text = shuffled_event
        
        if '_opg_' in task:
            task_inst = '\n'+row['blind_instruction_opg']
            answer_format = '\n' + row['answer_format_opg']
            question_text = task_inst + shuffled_text + answer_format + answer_str
        else:
            task_inst = '\n'+row['blind_instruction_opt']
            answer_format = '\n' + row['answer_format_opt'] 
            option_text = '\n' +row['options']+"."
            question_text = task_inst + shuffled_text+ answer_format + option_text + answer_str
    
    else:
        print("Task not supported")
        sys.exit()
    
    return question_text

def main():
    args = parse_args()
    
    print("############################################")
    print("Data Source: ", args.data_source)
    print("Task: ", args.task)
    print("Model ID: ", args.model_id)
    print("Date: ", pd.to_datetime('today').strftime("%Y-%m-%d"))
    print("############################################")
    
    larger_models = ['72b', '78b', '110b']
    if any(size in args.model_id.lower() for size in larger_models):
        load_8bit = True
        suffix = '8bit'
    else:
        load_8bit = False
        suffix = 'full'
        # load_8bit = True
        # suffix = '8bit'
    
    print("Loading 8-bit model: ", load_8bit)
    
    torch.cuda.empty_cache()
    
    if args.model_id == 'microsoft/Phi-3.5-mini-instruct':
        model = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        device_map="cuda", 
        torch_dtype="auto", 
        trust_remote_code=True, 
    )
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    elif args.model_id == 'Qwen/Qwen2.5-72B-Instruct':
        bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,  # Enable 8-bit loading
                bnb_4bit_use_double_quant=False,  # Optional: only needed for 4-bit quantization
                bnb_4bit_compute_dtype=torch.bfloat16  # Use bfloat16 for computation
            )
        
        model = AutoModelForCausalLM.from_pretrained(
                args.model_id, 
                device_map="cuda", 
                torch_dtype="auto", 
                trust_remote_code=True, 
                quantization_config=bnb_config,
            )
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        
         
    elif args.model_id == 'meta-llama/Llama-3.1-8B':
        pipe = pipeline("text-generation", model=args.model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
        
        
    else:
        print("Model not supported")
        sys.exit()

    if 'ordering_' in args.task:
        df = pd.read_pickle(f'data/benchmark/{args.data_source}_{args.task[:-10]}_test.pkl')
    else:
        df = pd.read_pickle(f'data/benchmark/{args.data_source}_{args.task}_test.pkl')
        
    df['preds'] = ''
    
    print("Making Inference for total number of samples: ", len(df))
    for index, row in df.iterrows():
        torch.cuda.empty_cache()
        question_text = process_benchmark(row, args.task)
        
        if args.model_id == 'microsoft/Phi-3.5-mini-instruct':
            messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": question_text},
        ]

            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )

            generation_args = {
                "max_new_tokens": 512,
                "return_full_text": False,
                "temperature": 0.0,
                "do_sample": False,
            }

            output = pipe(messages, **generation_args)
            pred = output[0]['generated_text']
        
        elif args.model_id == 'Qwen/Qwen2.5-72B-Instruct':
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": question_text}
            ]
            
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            pred = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
        elif args.model_id == 'meta-llama/Llama-3.1-8B':
            pred = pipe(question_text)[0]['generated_text'] 
  
        else:
            print("Model not supported")
            sys.exit()
        
        df.at[index, 'preds'] = pred
        
        
        if index % 300 == 0:
            print("Index: ", index)
            print(question_text)
            print("PR: ", pred)
            if 'opg' in args.task:
                print("GT: ", row['gt_answer'])
            else:
                print("GT: ", row['gt_option'])
            print("--------------------------------------------")
        
        
        torch.cuda.empty_cache()
        
    save_dir = f'data/predictions/blind/{args.model_id.split("/")[-1]}'
    os.makedirs(save_dir, exist_ok=True)

    df.to_pickle(f'{save_dir}/{args.data_source}_{args.task}_preds_{suffix}.pkl')
    
    
if __name__ == "__main__":
    main()
    
