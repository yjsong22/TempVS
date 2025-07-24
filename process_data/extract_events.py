import pandas as pd 
import openai
from openai import OpenAI
import spacy
import json
import time
from datetime import datetime, timedelta
import json

API_KEY = "YOUR_API_KEY" 
openai.api_key = API_KEY
client = OpenAI(api_key=API_KEY)
nlp = spacy.load('en_core_web_sm')

def is_name_or_proper_noun(sentence):
    character_names = ['pororo', 'loopy', 'eddy', 'harry', 'poby', 'tongtong', 'crong', 'rody', 'petty', 'wilma', 'fred', 'betty', 'barney', 'dino', 'pebbles', 'mr slate', 'baby puss', 'hoppy']
    
    doc = nlp(sentence)
    first_token = doc[0]
    
    if first_token.text.lower() in character_names:
        return True

    if first_token.ent_type_ == 'PERSON' or first_token.pos_ == 'PROPN':
        return True
    else:
        return False
    
def has_no_event(row):
    for col in row.index:
        if col.startswith("event") and row[col] == 'NO_EVENT':
            return True
    return False

# {"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 150}}
# {"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4", "messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 150}}

def extract_events_in_batch(df, data_source = 'pororo', chunk_size=10, time_limit=60):    
    batch_requests = []
    
    text_columns = [col for col in df.columns if col.startswith('text')]
    for col in text_columns:
        event_col = col.replace('text', 'event')
        if event_col not in df.columns:
            df[event_col] = None
    
    for idx, row in df.iterrows():
        for col in text_columns:
            if pd.notna(row[col]):
                custom_id = f"request_{row['story_id']}_{idx}_{col}"
                request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4",
                        "messages": [
                            {
                                "role": "user",
                                "content": f"""
                                    Given a text: {row[col]} Extract a single, concise and clear event sentence from the provided text. Ensure the sentence satisfies the following criteria: 
                                    1. The sentence should contain the event itself without phrases such as "the event is" or "event:". 
                                    2. If there are multiple events, extract only a single major event. 
                                    3. The sentence must contain only one clause with only one verb. 
                                    4. The event should be expressed in simple past tense.
                                    5. If no event is detected, return 'NO_EVENT'.
                                """
                            }
                        ],
                        "max_tokens": 150
                    }
                }
                batch_requests.append(request)
    
    all_responses = {}
    for chunk_id, i in enumerate(range(0, len(batch_requests), chunk_size)):
        chunk = batch_requests[i:i + chunk_size]
        
        with open(f'data/temporal_data/{data_source}/{data_source}_batch_requests_{chunk_id}.jsonl', 'w') as f:
            for request in chunk:
                f.write(json.dumps(request) + '\n')
        f.close()
                
        batch_input_file = client.files.create(file=open(f"data/temporal_data/{data_source}/{data_source}_batch_requests_{chunk_id}.jsonl", "rb"),purpose="batch")
        batch_input_file_id = batch_input_file.id
        batch_response = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"{data_source} event extraction job batch test"})
        batch_id = batch_response.id
        print(f"Batch {batch_id} created!")
            
        timeout = datetime.now() + timedelta(minutes=time_limit)
        
        while datetime.now() < timeout:
            batch_status = client.batches.retrieve(batch_id).status
            if batch_status == 'completed':
                file_response = client.files.content(client.batches.retrieve(batch_id).output_file_id).text
                with open(f"data/temporal_data/{data_source}/{data_source}_batch_responses_{chunk_id}.jsonl", "w") as f:
                    f.write(file_response)
                
                with open(f"data/temporal_data/{data_source}/{data_source}_batch_responses_{chunk_id}.jsonl", "r") as f:
                    for line in f:
                        response = json.loads(line)
                        response_id = response['custom_id'] #f"{row['story_id']}_request_{idx}_{col}" 26888_request_2_text0
                        extracted_event = response['response']['body']['choices'][0]['message']['content']
                        
                        _, _, idx, text_col = response_id.split('_')
                        idx = int(idx)
                        event_col = text_col.replace('text', 'event')
                        df.at[idx, event_col] = extracted_event.strip('"').strip("'")
                
                all_responses[response_id] = response 
                                    
                print("Batch completed and responses saved!")

                break
            
            elif batch_status == 'failed':
                print(f"Batch failed. Current status: {batch_status}")
                print(f"Error message: {client.batches.retrieve(batch_id).errors}")
                break
            else:
                print(f"Batch not completed. Current status: {batch_status}")
                time.sleep(30)  
        else:
            print(f"Timeout reached: Batch did not complete within {time_limit} minutes")
    
    df = df[~df.apply(has_no_event, axis=1)].reset_index(drop=True)

    with open(f"data/temporal_data/{data_source}/{data_source}_responses_all.jsonl", "w") as f:
        for response_id, response in all_responses.items():
            f.write(json.dumps(response) + '\n')
            
    return df
        