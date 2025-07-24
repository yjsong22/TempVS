import pandas as pd
import base64
import os
import openai
from openai import OpenAI
import random
import argparse
from PIL import Image
import base64
from io import BytesIO
import json
import sys


API_KEY = "YOUR_API_KEY" 
openai.api_key = API_KEY
client = OpenAI(api_key=API_KEY)


def encode_image(image):
    if isinstance(image, str):
        image = Image.open(image)
        
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes)
    img_str = img_base64.decode('utf-8')
    
    return img_str

def encode_image_gpt4v(image):
    return 'data:image/jpeg;base64,' + encode_image(image)

def find_characters_in_string(input_string, subset):
    with open(f"characters_{subset}.json", "r") as file:
        character_dict = json.load(file) 
    input_string_lower = input_string.lower()
    matches = [description for key, description in character_dict.items() if key in input_string_lower]
    return matches

def select_from_events_based_on_image(image_file_path, event_texts, subset, detail = 'auto', max_tokens=300):
    """ Uses GPT-4o to do the pretest """
    
    if subset in ['pororo', 'flintstones']:
        char_desp = " ".join(list(set(find_characters_in_string(event_texts[0], subset) + find_characters_in_string(event_texts[1], subset))))
        prompt = f"Below are the descriptions of the characters shown in the images: {char_desp}. Based on the image provided, select the event that best matches it from the following options: (A) {event_texts[0]} (B) {event_texts[1]} (C) None of the events. (D) Both events. Respond with only the corresponding letter: A, B, C, or D."
    else:
        prompt = f"Based on the image provided, select the event that best matches it from the following options: (A) {event_texts[0]} (B) {event_texts[1]} (C) None of the events. (D) Both events. Respond with only the corresponding letter: A, B, C, or D."
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [ {  "type": "text", "text": prompt }] \
                    + [ { "type": "image_url", "image_url": { "url": encode_image_gpt4v(image_file_path), "detail": detail } } ]
            }
        ],
        max_tokens=max_tokens,
    )
    
    return response.choices[0].message.content

def select_from_images_based_on_event(image_file_paths, event_text, event_texts, subset, detail = 'auto', max_tokens=300):
    if subset in ['pororo', 'flintstones']:
        char_desp = " ".join(list(set(find_characters_in_string(event_texts[0], subset) + find_characters_in_string(event_texts[1], subset))))
        prompt = f"Below are the descriptions of the characters shown in the images: {char_desp}. The event text is: {event_text}. Based on the event described, select the image that best matches it from the following options: (A) The first image. (B) The second image. (C) None of the images. (D) Both images. Respond with only the corresponding letter: A, B, C, or D."
    else:
        prompt = f"The event text is: {event_text}. Based on the event described, select the image that best matches it from the following options: (A) The first image. (B) The second image. (C) None of the images. (D) Both images. Respond with only the corresponding letter: A, B, C, or D."

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": [ {  "type": "text", "text": prompt }] \
                    + [ {"type": "image_url", "image_url": {"url": encode_image_gpt4v(image_fn), "detail": detail}} for i, image_fn in enumerate(image_file_paths)]
            }
        ],
        max_tokens=max_tokens,
    )
    
    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser(description="Perform grounding test")
    parser.add_argument("--subset", type=str, default="pororo", help="Choose the subset to test")
    parser.add_argument("--data_dir", type=str, default="/data/", help="Directory for data storage")
    parser.add_argument("--output_fn", type=str, default="grounding_test_predictions.txt", help="Prediction output file name")
    parser.add_argument("--max_tokens", type=int, default=300, help="Maximum tokens for GPT-4o API")
    parser.add_argument("--vision_detail", type=str, default="high", help="Detail level for GPT-4o API: auto, low, high")
    
    args = parser.parse_args()
    
    question_file_path = os.path.join(args.data_dir, args.subset, f"filtered_{args.subset}_questions.pkl")
    df_questions = pd.read_pickle(question_file_path)
    
    print(df_questions.shape)
    
    for _, row in df_questions.iterrows():
        for dist in range(1, 5):
            print(f"Distance: {dist}")
            event_pairs = row[f'pairs_dist_{dist}'].split('-')
            image_file_paths = [row[event_pairs[0].replace('event', 'link')], row[event_pairs[1].replace('event', 'link')]]
            event_texts = [row[event_pairs[0]], row[event_pairs[1]]]
            
            for image_file_path in image_file_paths:
                print("====================================")
                print(image_file_path)
                print(f'A:{event_texts[0]}, B:{event_texts[1]}')
                print(select_from_events_based_on_image(image_file_path, event_texts, subset=args.subset, detail=args.vision_detail, max_tokens=args.max_tokens))
            
            for event_text in event_texts:
                print("====================================")
                print(event_text)
                print(f'1st:{image_file_paths[0]} \n2nd:{image_file_paths[1]}')
                print(select_from_images_based_on_event(image_file_paths, event_text, event_texts, subset=args.subset, detail=args.vision_detail, max_tokens=args.max_tokens))
            
    


    
if __name__ == "__main__":
    main()