from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from longva.model.builder import load_pretrained_model
from longva.mm_utils import tokenizer_image_token, process_images
from longva.constants import IMAGE_TOKEN_INDEX
from PIL import Image
from decord import VideoReader, cpu
import torch
import numpy as np
# fix seed
torch.manual_seed(0)



def answer_question(
    image_paths, question_text, image_processor, model, tokenizer,  
    device = "cuda:0",  max_new_tokens = 512
):
    gen_kwargs = {"do_sample": False, "temperature": 0.0, "top_p": None, "num_beams": 1, "use_cache": True, "max_new_tokens": max_new_tokens}
    #tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, "llava_qwen", device_map="cuda:0")


    prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{question_text}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    image = [Image.open(img_path).convert("RGB") for img_path in image_paths]
    images_tensor = process_images(image, image_processor, model.config)
    images_tensor = [_image.to(dtype=torch.float16, device=device) for _image in images_tensor]

    with torch.inference_mode():
        output_ids = model.generate(input_ids, images=images_tensor, image_sizes=[img.size for img in image], modalities=["video"], **gen_kwargs)
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs