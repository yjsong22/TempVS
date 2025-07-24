from data_filters import *
from extract_events import *
import pandas as pd
import os
import argparse
import sys

def capitalize_names(sentence):
    pororo_names = ['pororo', 'loopy', 'eddy', 'harry', 'tongtong', 'crong', 'rody', 'petty', 'pipi','popo', 'poby', 'toto', 'tutu']
    words = sentence.split()
    
    capitalized_sentence = ' '.join([word.capitalize() if word.lower() in  pororo_names else word for word in words])
    
    return capitalized_sentence

def load_pororo_data(data_dir, use_original_color=False):
    all_data_pororo = []
    
    if use_original_color:
        print("Load original color image dataset ...")
        image_dir = os.path.join(data_dir, 'original_color', 'PororoSV', 'full', 'images')
        annot_fn = os.path.join(data_dir, 'original_color', 'PororoSV', 'full', 'full.json')
        with open(annot_fn, 'r') as file:
            annotations = json.load(file)['data']
            
        for i in range(len(annotations)):
            annot = annotations[i]
            context = annot['task_instance']['context']
            captions = re.findall(r'Caption#\d+: (.*?)(?={image#|$)', context) + [annot['response']]
            texts = [caption.strip().lower() for caption in captions]  
            texts = clean_repeated_parts(texts)
            img_paths = [os.path.join(image_dir, img_id) for img_id in annot['task_instance']['images_path']]
            
            if len(texts) != 5 or len(img_paths) != 5:
                #print(f"Skipping sample {annot['sample_id']} with {len(texts)} texts and {len(img_paths)} images")
                continue
            assert len(texts) == 5
            assert len(img_paths) == 5
            
            row_data = {
                'img_seq_id': f"full_{annot['sample_id']}",
                'img_paths': img_paths,
                'full_text': ' '.join(texts),
                'split': 'full'
            }
            
            for i in range(5):
                row_data[f'link{i}'] = img_paths[i] if i < len(img_paths) else ''
                row_data[f'text{i}'] = capitalize_names(texts[i]) if i < len(texts) else ''
            
            all_data_pororo.append(row_data)
        
    else:
        descriptions = np.load(os.path.join(data_dir, 'descriptions.npy'), allow_pickle=True, encoding='latin1').item()
        imgs_list = np.load(os.path.join(data_dir, 'img_cache4.npy'), encoding='latin1')
        followings_list = np.load(os.path.join(data_dir, 'following_cache4.npy'))
        train_ids, val_ids, test_ids = np.load(os.path.join(data_dir, 'train_seen_unseen_ids.npy'), allow_pickle=True)
        train_ids = np.sort(train_ids)
        val_ids = np.sort(val_ids)
        test_ids = np.sort(test_ids)
        
        for subset, ids in {'train': train_ids, 'val': val_ids, 'test': test_ids}.items():
            for i, item in enumerate(ids):
                img_paths = [str(imgs_list[item])[2:-1]] + [str(followings_list[item][i])[2:-1] for i in range(4)]
                tgt_img_ids = [str(img_path).replace('.png', '') for img_path in img_paths]
                texts = [descriptions[tgt_img_id][0] for tgt_img_id in tgt_img_ids]
                texts = [text[0].upper() + text[1:] for text in texts]
                img_paths = [os.path.join(data_dir, img_path) for img_path in img_paths]
                assert len(texts) == 5
                assert len(img_paths) == 5
                
                row_data = {
                    'img_seq_id':f'{subset}_{item}',
                    'img_paths': img_paths,
                    'full_text': ' '.join(texts),
                    'split': subset
                }
                
                for i in range(5):
                    row_data[f'link{i}'] = img_paths[i] if i < len(img_paths) else ''
                    row_data[f'text{i}'] = capitalize_names(texts[i]) if i < len(texts) else ''
                
                all_data_pororo.append(row_data)

    return pd.DataFrame(all_data_pororo)

def main():
    parser = argparse.ArgumentParser(description="Process pororo data")
    parser.add_argument("--data_dir", type=str, default="data/temporal_data/pororo", help="Directory for data storage")
    parser.add_argument("--output_fn", type=str, default="filtered_pororo_events.pkl", help="Output file name")
    parser.add_argument("--img_seq_only_criteria", type=str, default='link0', help="How to determine distinguished image sequence?")
    parser.add_argument("--select_shortest", type=bool, default=True, help="Select shortest story if multiple are available")
    parser.add_argument("--use_original_color", type=bool, default=True, help="Use original color dataset")
    parser.add_argument("--clip_sim_threshold", type=float, default=0, help="Threshold for clip similarity between image and text")
    parser.add_argument("--filter_data", type=bool, default=True, help="Start with filtering data")
    parser.add_argument("--use_bertscore", type=bool, default=True, help="Create use bertscore for similarity calculation")
    parser.add_argument("--use_clipsim", type=bool, default=True, help="Use CLIP for similarity calculation")
    parser.add_argument("--text_sim_threshold", type=float, default=0.95, help="Similarity threshold for text filtering")
    parser.add_argument("--img_sim_threshold", type=float, default=0.94, help="Similarity threshold for image filtering")
    parser.add_argument("--text_thres_list", type=list, default=[0.98, 0.98, 0.96, 0.92, 0.92, 0.92], help="Similarity thresholds for bertscore's precsion, recall and f1")
    parser.add_argument("--filtered_data_fn", type=str, default='data/temporal_data/pororo/filtered_pororo.csv', help="Filtered data file before extracting events")
    
    args = parser.parse_args()
    
    
    if args.filter_data:
        df_pororo = load_pororo_data(args.data_dir, args.use_original_color)
        filtered_pororo = filter_data(df_pororo, args.img_sim_threshold, args.img_seq_only_criteria, args.select_shortest, args.clip_sim_threshold, args.use_bertscore, args.use_clipsim, args.text_sim_threshold, args.text_thres_list)
        
        print(args.img_sim_threshold)
        print(args.text_thres_list)
        print(f"Original pororo data: {len(df_pororo)}")
        print(f"Filtered pororo data: {len(filtered_pororo)}")
            
        filtered_pororo.to_csv(args.filtered_data_fn, index=False)
        print(f"Filtered pororo data saved to: {args.filtered_data_fn}")

        combine_images(filtered_pororo, 'pororo', args.data_dir)


    else:
        filtered_pororo = pd.read_csv(args.filtered_data_fn)
        
    
    print("Extract events ...")
    df_with_events = extract_events_in_batch(filtered_pororo, 'pororo', chunk_size=500)

    output_path = os.path.join(args.data_dir, args.output_fn)
    df_with_events.to_pickle(output_path)
    print(f"Filtered pororo data with events saved to: {output_path}")

    print("Create questions ...")
    df_with_questions = create_questions(df_with_events)
    df_with_questions.to_pickle(os.path.join(args.data_dir, args.output_fn.replace('events', 'questions')))
    
    
    
if __name__ == "__main__":
    main()
