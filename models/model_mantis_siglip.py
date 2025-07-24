from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 
from mantis.models.mllava import MLlavaProcessor, LlavaForConditionalGeneration

from mantis.models.mllava import chat_mllava
import torch


def answer_question(img_path, question_text, model, processor, max_new_tokens=512):
    images = [Image.open(path).convert("RGB") for path in img_path]

    
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_beams": 1,
        "do_sample": False
    }
    
    response, history = chat_mllava(question_text, images, model, processor, **generation_kwargs)
    
    return response
    