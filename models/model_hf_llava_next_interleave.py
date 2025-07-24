from transformers import pipeline, AutoProcessor
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True   
import warnings
warnings.filterwarnings("ignore")
import sys

def answer_question(img_path, question_text, model, processor, use_separate, max_new_tokens=512):
    images = [Image.open(path).convert("RGB") for path in img_path]
    
    if use_separate:
        image_tokens = [{"type": "image", "image_id": f"image_{chr(97 + i)}"} for i in range(len(img_path))]
    else:
        image_tokens = [{"type": "image", "image_id": f"image_{i+1}"} for i in range(len(img_path))]
    conversation = [
          {
        "role": "user",
        "content": [
            {"type": "text", "text": question_text},
            *image_tokens
            ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=images, text=prompt, return_tensors='pt').to(0, torch.float16)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(output[0][2:], skip_special_tokens=True)

