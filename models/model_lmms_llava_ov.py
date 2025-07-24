from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import copy
import torch

import sys
import warnings

warnings.filterwarnings("ignore")
# print(conv_templates.keys())
# # dict_keys(['default', 'v0', 'v1', 'vicuna_v1', 'llama_2', 'mistral_instruct', 'mistral_orca', 'mistral_zephyr', 'mistral_direct', 'plain', 'v0_plain', 'chatml_direct', 'llava_v0', 'llava_v0_mmtag', 'llava_v1', 'llava_v1_mmtag', 'llava_llama_2', 'llava_llama_3', 'llava_llama_2_simple', 'llava_llama_2_mmtag', 'llava_mistral_instruct', 'mpt', 'qwen_1_5', 'qwen_2', 'gemma_instruct'])


def answer_question(
    image_paths, question_text, image_processor, model, tokenizer,  
    device = "cuda", conv_template = "qwen_1_5", max_new_tokens = 512
):
    images = [Image.open(path).convert("RGB") for path in image_paths]
    
    image_tensor = process_images(images, image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
  
    conv = copy.deepcopy(conv_templates[conv_template])

    question = " ".join([DEFAULT_IMAGE_TOKEN for _ in range(len(images))]) + "\n" + question_text
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    
    input_ids = tokenizer_image_token(
        prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)
    image_sizes = [image.size for image in images]

    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=max_new_tokens,
    )

    # Decode the output
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    return text_outputs[0]  
