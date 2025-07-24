import torch
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  


def answer_question(img_path, question_text, model, processor, use_separate, max_new_tokens=512):
    if len(img_path) > 1:
        raw_image = [Image.open(path).convert("RGB").resize((224, 224), Image.Resampling.LANCZOS) for path in img_path]
    else:
        raw_image = [Image.open(path).convert("RGB") for path in img_path]
    
    if use_separate:
        image_tokens = [{"type": "image", "image_id": f"image_{chr(97 + i)}"} for i in range(len(img_path))]
    else:    
        image_tokens = [{"type": "image", "image_id": f"image_{i}"} for i in range(len(img_path))]

    conversation = [
        {
        "role": "user",
        "content": [
            {"type": "text", "text": question_text},
            *image_tokens,
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    inputs_image = processor(text=prompt, images=raw_image, return_tensors='pt').to(0, torch.float16)

    output = model.generate(**inputs_image, max_new_tokens=max_new_tokens, do_sample=False)
    
    return processor.decode(output[0][2:], skip_special_tokens=True)
