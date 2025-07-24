from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 

def answer_question(img_path, question_text, model, processor, max_new_tokens=512):  
    images = []
    placeholder = ""

    for i in range(len(img_path)):
        images.append(Image.open(img_path[i]).convert("RGB"))
        placeholder += f"<|image_{i+1}|>\n"

    messages = [
        {"role": "user", "content": placeholder+question_text},
    ]

    prompt = processor.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    inputs = processor(prompt, images, return_tensors="pt").to("cuda:0") 

    generation_args = { 
        "max_new_tokens": max_new_tokens, 
        "temperature": 0.0, 
        "do_sample": False, 
    } 

    generate_ids = model.generate(
        **inputs, 
        eos_token_id=processor.tokenizer.eos_token_id, 
        **generation_args
    )

    # remove input tokens 
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(
        generate_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0] 

    return response

def batch_answer_questions(batch_img_paths, batch_questions, model, processor_base):
    """
    Process a batch of questions and images for Phi Vision model
    
    Args:
        batch_img_paths: List of lists of image paths
        batch_questions: List of question texts
        model: Phi Vision model
        processor_base: Base processor to clone with appropriate num_crops
        
    Returns:
        List of model predictions
    """
    batch_size = len(batch_questions)
    batch_predictions = []
    
    # Process in smaller sub-batches to avoid OOM
    sub_batch_size = 4  # Adjust based on GPU memory
    
    for i in range(0, batch_size, sub_batch_size):
        end_idx = min(i + sub_batch_size, batch_size)
        sub_batch_img_paths = batch_img_paths[i:end_idx]
        sub_batch_questions = batch_questions[i:end_idx]
        
        # Process each sample in the sub-batch individually
        sub_batch_preds = []
        for img_paths, question in zip(sub_batch_img_paths, sub_batch_questions):
            # Create a processor with the correct number of crops for this sample
            processor = AutoProcessor.from_pretrained(
                processor_base.name_or_path,
                trust_remote_code=True,
                num_crops=len(img_paths)
            )
            
            # Load and process images
            images = [Image.open(img_path).convert('RGB') for img_path in img_paths]
            
            # Process inputs
            inputs = processor(text=question, images=images, return_tensors="pt").to("cuda")
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            
            # Decode response
            response = processor.decode(outputs[0], skip_special_tokens=True)
            
            if "The answer is:" in response:
                answer = response.split("The answer is:")[-1].strip()
            else:
                answer = response
                
            sub_batch_preds.append(answer)
        
        batch_predictions.extend(sub_batch_preds)
    
    return batch_predictions

