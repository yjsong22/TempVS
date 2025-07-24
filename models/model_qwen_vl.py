from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import warnings
warnings.filterwarnings("ignore")

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



def answer_question(img_path, question_text, model, processor, use_separate=False, max_new_tokens=512):

    content = []
    if use_separate:
        for i in range(len(img_path)):
            content.append({
                "type": "image", "image": img_path[i], "image_id": f"image_{chr(97 + i)}"  
            })
    
    else:
        for i in range(len(img_path)):
            content.append({
                "type": "image","image": img_path[i], "image_id": f"image_{i+1}"
            })
    content.append({"type": "text", "text": question_text})
    
    messages = [{
        "role": "user",
        "content": content
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]
