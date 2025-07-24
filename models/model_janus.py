from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from transformers import AutoModelForCausalLM
from janus.utils.io import load_pil_images

def answer_question(img_path, question_text, vl_gpt, vl_chat_processor, tokenizer, use_separate, max_new_tokens=512):
    image_content = ""
    if len(img_path) == 1:
        image_content = "<image_placeholder>\n"
        
    elif len(img_path) == 2:
        image_content = "This is the First image: <image_placeholder>\nThis is the Second image: <image_placeholder>\n"
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
        

    # load images and prepare for inputs
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(vl_gpt.device)

    # # run image encoder to get the image embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # # run the model to get the response
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True,
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    
    return answer
    