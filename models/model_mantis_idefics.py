import requests
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from io import BytesIO

from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image


def answer_question(img_path, question_text, model, processor, use_separate, max_new_tokens=512):
    images = [Image.open(path).convert('RGB') for path in img_path]
    if use_separate:
        image_tokens = [{"type": "image", "image_id": f"image_{chr(97 + i)}"} for i in range(len(img_path))]
    else:
        image_tokens = [{"type": "image", "image_id": f"image_{i}"} for i in range(len(img_path))]
    
    generation_kwargs = {
    "max_new_tokens": max_new_tokens,
    "num_beams": 1,
    "do_sample": False
}
    messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": question_text},
            *image_tokens,
        ]
    }    
]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=images, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    generated_ids = model.generate(**inputs, **generation_kwargs)
    response = processor.batch_decode(generated_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response[0]
    
    
    