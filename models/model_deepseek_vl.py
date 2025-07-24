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
        
    
    
    
def main():
    model_path = "deepseek-ai/deepseek-vl2-tiny"
    vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    
    question_text = "This is a description of the possible appearance of character(s) in the image(s): Fred is chubby, has black hair, a large nose and wears an orange and black spotted short-sleeved loincloth with a blue scarf. Wilma has swirly red-orange hair, wears a loincloth dress with a torn hemline and a pearl necklace. \n The statement is: Fred and wilma talked to each other in the room after Fred talked in the living room while wearing a head and arm bandage. \n Is the statement completely accurate and content with the content in the sequence of images? When making the choice, focus on the evidence presented in the sequence of images from left to right. No need to give reasoning process. Submit only the right option letter as your answer, e.g., Option [Letter]. Options: A. True; B. False. The answer is: "
    img_path = ['/home/ysong/temporal-mllms/data/temporal_data/flintstones/video_frames_sampled_jpg/s_01_e_01_shot_013354_013428.jpg',
 '/home/ysong/temporal-mllms/data/temporal_data/flintstones/video_frames_sampled_jpg/s_01_e_01_shot_014003_014077.jpg',
 '/home/ysong/temporal-mllms/data/temporal_data/flintstones/video_frames_sampled_jpg/s_01_e_01_shot_014586_014660.jpg',
 '/home/ysong/temporal-mllms/data/temporal_data/flintstones/video_frames_sampled_jpg/s_01_e_01_shot_014784_014858.jpg',
 '/home/ysong/temporal-mllms/data/temporal_data/flintstones/video_frames_sampled_jpg/s_01_e_01_shot_015224_015298.jpg']
    print(question_text)
    answer = answer_question(img_path, question_text, vl_gpt, vl_chat_processor, tokenizer, use_separate=True)
    print(answer)
    
    
   
if __name__ == "__main__":
    main()




