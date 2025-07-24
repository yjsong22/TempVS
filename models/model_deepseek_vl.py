import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

def answer_question(img_path, question_text, vl_gpt, vl_chat_processor, tokenizer, use_separate = False, max_new_tokens=512):
    image_content = ""
    if len(img_path) == 1:
        image_content = "<image>\n"
        
    elif len(img_path) == 2:
        image_content = "This is the First image: <image>\nThis is the Second image: <image>\n"
    else:
        if use_separate:
            for i in range(len(img_path)):
                image_content += f"This is Image_{chr(97 + i)}: <image>\n"
        else:
            for i in range(len(img_path)):
                image_content += f"This is image_{i+1}: <image>\n"
    
    conversation = [
        {
            "role": "<|User|>",
            "content": image_content + question_text,
            "images": img_path,
        },
        {"role": "<|Assistant|>", "content": ""}
    ]


    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    ).to(vl_gpt.device)

    with torch.no_grad():
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        inputs_embeds, past_key_values = vl_gpt.incremental_prefilling(
            input_ids=prepare_inputs.input_ids,
            images=prepare_inputs.images,
            images_seq_mask=prepare_inputs.images_seq_mask,
            images_spatial_crop=prepare_inputs.images_spatial_crop,
            attention_mask=prepare_inputs.attention_mask,
            chunk_size=512 # prefilling size
        )

        # run the model to get the response
        outputs = vl_gpt.generate(
            inputs_embeds=inputs_embeds,
            input_ids=prepare_inputs.input_ids,
            images=prepare_inputs.images,
            images_seq_mask=prepare_inputs.images_seq_mask,
            images_spatial_crop=prepare_inputs.images_spatial_crop,
            attention_mask=prepare_inputs.attention_mask,
            past_key_values=past_key_values,

            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,

            do_sample=False,
            use_cache=True,
        )

        answer = tokenizer.decode(outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(), skip_special_tokens=False)
        #print(f"{prepare_inputs['sft_format'][0]}", answer)
        return answer
        
    
    





