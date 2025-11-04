import os
from matplotlib import path, pyplot as plt
import pandas as pd
import argparse
from tqdm import tqdm
import sys
import torch

from llava.model.builder import load_pretrained_model
from transformers import AutoProcessor, AutoModelForCausalLM, AutoModel, AutoTokenizer, AutoModelForVision2Seq
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration, LlavaForConditionalGeneration
from transformers import Qwen2VLForConditionalGeneration
from transformers import BitsAndBytesConfig
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM

import models.model_hf_llava_next_interleave as model_hf_llava_next_interleave
import models.model_lmms_llava_ov as model_lmms_llava_ov
import models.model_qwen_vl as model_qwen_vl
import models.model_phi_vision as model_phi_vision
import models.model_intern_vl as model_intern_vl
import models.model_hf_llava_video as model_hf_llava_video
import models.model_mantis_siglip as model_mantis_siglip
import models.model_mantis_idefics as model_mantis_idefics
import models.model_longva as model_longva
import models.model_deepseek_vl as model_deepseek_vl

import warnings
warnings.filterwarnings("ignore")

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference')
    parser.add_argument('--data_source', type=str, default='flintstones',
                        help='Source of the data (e.g., flintstones, pororo, vist, vwp)')
    parser.add_argument('--task', type=str, default='single_grounding_paired',
                        choices=[ 
                                 'paired_event_discrimination', 'single_grounding_paired',
                                 'triple_event_discrimination', 'single_grounding_triple',
                                 'single_grounding_all', 'single_grounding_all_story', 
                                 'ordering_images_opt_story',  'ordering_images_opt_event',
                                 'ordering_texts_opt_story', 'ordering_texts_opt_event'
                                 ],
                        help='Type of task to perform')
    parser.add_argument('--model_id', type=str, 
                        default= 'lmms-lab/llava-onevision-qwen2-72b-ov-sft', 
                        help='Pretrained model path')
    parser.add_argument('--model_name', type=str, default='llava_qwen',
                        help='Model name')
    parser.add_argument('--run_cot', type=bool, default=False, help='Run model with Chain of Thought')

    return parser.parse_args()


def process_benchmark(row, task, data_source, run_cot=False, use_separate=False):
    char_desp = 'This is a description of the possible appearance of character(s) in the image(s): ' + row['character_info']
    task_inst = '\n'+row['task_instruction']
    if run_cot and 'event_discrimination' in task:
        # print("COT for event discrimination task")
        task_inst = '\n'+row['task_instruction_cot']
    if run_cot and 'ordering' in task:
        # print("COT for ordering task")
        task_inst = '\n'+row['task_instruction_cot_opt']
    
    
    answer_str = '\nThe answer is: '
    
    if data_source in ['vist', 'vwp']:
        char_desp = ''


    if task in ['paired_event_discrimination', 'triple_event_discrimination', 'ordering_images_opt_story',  'ordering_images_opt_event', 'ordering_texts_opt_story', 'ordering_texts_opt_event']:
        option_text = '\nOptions are: '+row['option_text']

    if 'grounding' in task:
        if use_separate:
            img_path = row['img_path_list']
        else:
            img_path = [row['combined_img_path']]
        if 'story' in task:
            event_text ='\nThe event is: '+row['text_to_test']   
        else:
            event_text ='\nThe event is: '+row['event_to_test']   
        question_text = char_desp + event_text + task_inst + answer_str
        
    elif 'event_discrimination' in task:
        if use_separate:
            img_path = row['img_path_list']
        else:
            img_path = [row['combined_img_path']]
        statement = '\nThe statement is: '+row['statement_text']
        question_text = char_desp + statement + task_inst + option_text + answer_str        
        
    elif 'ordering_images' in task:
        img_index_str = "The images I provide are labeled in order as Image a, Image b, Image c, Image d, and Image e, and so on if there are more photos."
        if use_separate:
            img_path = row['img_path_list_shuffled']
        else:
            img_path = [row['combined_img_path_shuffled']]
        story_text ='\nThe story is: ' + row['texts_gt']
        event_text = '\nThe events are: ' + row['events_gt']
        
        if 'story' in task:
            gt_text = story_text
        else:
            gt_text = event_text
        
        if '_opg_' in task:
            answer_format = '\n' + row['answer_format_opg']
            question_text = char_desp + gt_text + task_inst + img_index_str + answer_format + answer_str
        else: #'_opt_' in task
            answer_format = '\n' + row['answer_format_opt'] 
            option_text = '\n' +row['options']+"."
            question_text = char_desp + gt_text + task_inst + img_index_str + answer_format + option_text + answer_str
    
    elif 'ordering_texts' in task:
        if use_separate:
            img_path = row['img_path_list_gt']
        else:
            img_path = [row['combined_img_path_gt']]
        shuffled_story = '\nThe shuffled sentences are:\n' + '\n'.join(f"Sentence {chr(97 + i)}: {text}" for i, text in enumerate(row['texts_shuffled'])) 
        shuffled_event = '\nThe shuffled events are:\n' + '\n'.join(f"Sentence {chr(97 + i)}: {event}" for i, event in enumerate(row['events_shuffled']))  
        
        if 'story' in task:
            shuffled_text = shuffled_story
        else:
            shuffled_text = shuffled_event
        
        if '_opg_' in task:
            answer_format = '\n' + row['answer_format_opg']
            question_text = char_desp + shuffled_text + task_inst + answer_format + answer_str
        else:
            answer_format = '\n' + row['answer_format_opt'] 
            option_text = '\n' +row['options']+"."
            question_text = char_desp + shuffled_text + task_inst + answer_format + option_text + answer_str
    
    else:
        print("Task not supported")
        sys.exit()
    
    return img_path, question_text

def main():
    args = parse_args()
    args.use_separate = False
    
    print("############################################")
    print("Data Source: ", args.data_source)
    print("Task: ", args.task)
    print("Model ID: ", args.model_id)
    if args.use_separate:
        print("Seperate Images")
    else:
        print("Combined Images")
    print("Running CoT: ", args.run_cot)
    print("Date: ", pd.to_datetime('today').strftime("%Y-%m-%d"))
    print("############################################")
    
    larger_models = ['72b', '78b', '110b']
    if any(size in args.model_id.lower() for size in larger_models):
        load_8bit = True
        suffix = '8bit'
    else:
        load_8bit = False
        suffix = 'full'
    
    print("Loading 8-bit model: ", load_8bit)
    
    torch.cuda.empty_cache()
    if 'lmms-lab/llava-onevision' in args.model_id:
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            args.model_id, None, args.model_name, load_8bit=load_8bit, device_map="auto",
        )
        model.eval() 
    elif 'Qwen/Qwen2-VL' in args.model_id or 'Qwen/Qwen2.5-VL' in args.model_id:
        bnb_config = None
        # if 'Qwen/Qwen2-VL-72B' in args.model_id:
        #      bnb_config = BitsAndBytesConfig(
        #         load_in_4bit=True,  # Changed from 8-bit to 4-bit
        #         bnb_4bit_use_double_quant=True,  # Enable double quantization
        #         bnb_4bit_quant_type="nf4",  # Use normal float 4 format
        #         bnb_4bit_compute_dtype=torch.bfloat16  # Use bfloat16 for computation
        #     )
        # else:
        if load_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,  # Enable 8-bit loading
                bnb_4bit_use_double_quant=False,  # Optional: only needed for 4-bit quantization
                bnb_4bit_compute_dtype=torch.bfloat16  # Use bfloat16 for computation
            )
            

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            quantization_config=bnb_config if load_8bit else None,  
        ).eval()

        processor = AutoProcessor.from_pretrained(args.model_id)
        
    elif 'llava-hf/llava-next' in args.model_id or 'llava-hf/llava-interleave' in args.model_id:
        from transformers import LlavaForConditionalGeneration
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
        ).to(0)
        processor = AutoProcessor.from_pretrained(args.model_id)
        
    elif 'llava-hf/LLaVA-NeXT-Video' in args.model_id:
        bnb_config = None
        if load_8bit:
            bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,  # Enable 8-bit loading
                    bnb_4bit_use_double_quant=False,  # Optional: only needed for 4-bit quantization
                    bnb_4bit_compute_dtype=torch.bfloat16  # Use bfloat16 for computation
                )
        
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        args.model_id, 
        torch_dtype=torch.float16, 
        attn_implementation="flash_attention_2",
        device_map="auto",
        quantization_config=bnb_config if load_8bit else None).eval()
        
        processor = LlavaNextVideoProcessor.from_pretrained(args.model_id)
    
    elif 'lmms-lab/LongVA' in args.model_id:
        tokenizer, model, image_processor, _ = model_longva.load_pretrained_model(args.model_id, None, "llava_qwen", device_map="cuda:0")
    
    elif 'microsoft/Phi' in args.model_id:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, 
            device_map="cuda", 
            trust_remote_code=True, 
            torch_dtype="auto", 
            _attn_implementation='flash_attention_2'    
            ).eval()
        
    elif 'OpenGVLab/InternVL' in args.model_id:            
        model = AutoModel.from_pretrained(
                args.model_id,
                torch_dtype=torch.bfloat16,
                load_in_8bit=load_8bit,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True).eval()
        if not load_8bit:
            model = model.cuda()
            
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, use_fast=False)
    
    elif 'TIGER-Lab/Mantis-8B-siglip-llama3' in args.model_id:
        from mantis.models.mllava import MLlavaProcessor, LlavaForConditionalGeneration
        model = LlavaForConditionalGeneration.from_pretrained(args.model_id, device_map="cuda", torch_dtype=torch.bfloat16, attn_implementation= "flash_attention_2", trust_remote_code=True)
        processor = MLlavaProcessor.from_pretrained(args.model_id)
    
    elif 'TIGER-Lab/Mantis-8B-Idefics2' in args.model_id:
        model = AutoModelForVision2Seq.from_pretrained(
        args.model_id,
        device_map="cuda"
    )
        processor = AutoProcessor.from_pretrained(args.model_id, do_image_splitting = False) # do_image_splitting is False by default
        
    elif 'deepseek-ai/deepseek' in args.model_id:
        vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(args.model_id)
        tokenizer = vl_chat_processor.tokenizer

        vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True)
        vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
        
    elif 'deepseek-ai/Janus' in args.model_id:
        if 'Flow' in args.model_id:
            from janus.janusflow.models import MultiModalityCausalLM, VLChatProcessor
            vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.model_id)
            tokenizer = vl_chat_processor.tokenizer

            vl_gpt = MultiModalityCausalLM.from_pretrained(
            args.model_id, trust_remote_code=True
        )
            vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
            
        else:
            from janus.models import MultiModalityCausalLM, VLChatProcessor
            vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.model_id)
            tokenizer = vl_chat_processor.tokenizer

            vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
                args.model_id, trust_remote_code=True
            )
            vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
        
        
    else:
        print("Model not supported")
        sys.exit()

    if 'ordering_' in args.task:
        df = pd.read_pickle(f'data/benchmark_pkl/{args.data_source}_{args.task[:-10]}_test.pkl')
    elif args.task == 'single_grounding_all_story':
        df = pd.read_pickle(f'data/benchmark_pkl/{args.data_source}_{args.task[:-6]}_test.pkl')
    else:
        df = pd.read_pickle(f'data/benchmark_pkl/{args.data_source}_{args.task}_test.pkl')
        
    df['preds'] = ''
    print("Making Inference for total number of samples: ", len(df))
    for index, row in df.iterrows():
        torch.cuda.empty_cache()
        img_path, question_text = process_benchmark(row, args.task, args.data_source,  args. run_cot, args.use_separate)
        if 'lmms-lab/llava-onevision' in args.model_id:
            pred = model_lmms_llava_ov.answer_question(img_path, question_text, image_processor, model, tokenizer)
       
        elif 'Qwen/Qwen2-VL' in args.model_id or 'Qwen/Qwen2.5-VL' in args.model_id:
            pred = model_qwen_vl.answer_question(img_path, question_text, model, processor, args.use_separate)
       
        elif 'llava-hf/llava-next' in args.model_id or 'llava-hf/llava-interleave' in args.model_id:
            pred = model_hf_llava_next_interleave.answer_question(img_path, question_text, model, processor, args.use_separate)
        
        elif 'llava-hf/LLaVA-NeXT-Video' in args.model_id:
            pred = model_hf_llava_video.answer_question(img_path, question_text, model, processor, args.use_separate)
            
        elif 'lmms-lab/LongVA' in args.model_id:
            pred = model_longva.answer_question(img_path, question_text, image_processor, model, tokenizer)
        
        elif 'microsoft/Phi' in args.model_id:
            processor = AutoProcessor.from_pretrained(args.model_id, 
                                                      trust_remote_code=True, 
                                                      num_crops=len(img_path),) 
            pred = model_phi_vision.answer_question(img_path, question_text, model, processor)
        elif 'OpenGVLab/InternVL' in args.model_id:
            if len(img_path) == 1:
                pixel_values = model_intern_vl.load_image(img_path, max_num=12).to(torch.bfloat16).cuda()
            else:
                pixel_list = [model_intern_vl.load_image(img, max_num=12).to(torch.bfloat16).cuda() for img in img_path]
                pixel_values = torch.cat(pixel_list, dim=0)
            generation_config = {
                'max_new_tokens': 512,
                'do_sample': True,
                'pad_token_id': tokenizer.eos_token_id,  
                'eos_token_id': tokenizer.eos_token_id  
                }
            pred = model.chat(tokenizer, pixel_values, '<image>\n'+question_text, generation_config)
       
        elif 'TIGER-Lab/Mantis-8B-siglip-llama3' in args.model_id:
            pred = model_mantis_siglip.answer_question(img_path, question_text, model, processor)
        
        elif 'TIGER-Lab/Mantis-8B-Idefics2' in args.model_id: 
            pred = model_mantis_idefics.answer_question(img_path, question_text, model, processor, args.use_separate)
            
        elif 'deepseek-ai/deepseek' in args.model_id:
            pred = model_deepseek_vl.answer_question(img_path, question_text, vl_gpt, vl_chat_processor, tokenizer, args.use_separate)
            
        elif 'deepseek-ai/Janus' in args.model_id:
            pred = model_janus.answer_question(img_path, question_text, vl_gpt, vl_chat_processor, tokenizer, use_separate=args.use_separate)
            
        else:
            print("Model not supported")
            sys.exit()
        
        df.at[index, 'preds'] = pred
    
        
        
        if index % 121 == 0 :
            print("Index: ", index)
            print(question_text)
            print(img_path)
            print("PR: ", pred)
            if 'opg' in args.task:
                print("GT: ", row['gt_answer'])
            else:
                print("GT: ", row['gt_option'])
            print("--------------------------------------------")
        
        torch.cuda.empty_cache()
        
        
    if args.run_cot:
        save_dir = f'data/predictions/cot/{args.model_id.split("/")[-1]}'
       
    else:
        save_dir = f'data/predictions/combined/{args.model_id.split("/")[-1]}'
    os.makedirs(save_dir, exist_ok=True)
     
 
    df.to_pickle(f'{save_dir}/{args.data_source}_{args.task}_preds_{suffix}.pkl')
        
    
    
if __name__ == "__main__":
    main()
    
