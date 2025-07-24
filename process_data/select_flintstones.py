from data_filters import *
from extract_events import *
import pandas as pd
import os
import argparse
import sys
from flintstones_dataloader import StoryImageDataset
import random


def convert_npy_to_jpg(img_id):
    img_np_path = f'data/temporal_data/flintstones/video_frames_sampled/{img_id}.npy'
    image_data = np.load(img_np_path)
    
    random_index = random.randint(0, image_data.shape[0] - 1)
    random_image = image_data[random_index]
    
    if random_image.dtype != np.uint8:
        random_image = (255 * (random_image - random_image.min()) / (random_image.max() - random_image.min())).astype(np.uint8)
        
    img = Image.fromarray(random_image)
    img.save(f'data/temporal_data/flintstones/video_frames_sampled_jpg/{img_id}.jpg', "JPEG")

def load_flintstones_data(data_dir):
    all_data_flintstones = []
    
    for split in ['train', 'val', 'test']:
        print(f"Processing {split} subset...")
        dataset = StoryImageDataset(data_dir, None, None, split)
        
        for item in range(len(dataset)):
            texts = []
            jpg_paths = [dataset.orders[item]] + dataset.followings[dataset.orders[item]]
            for idx, jpg_path in enumerate(jpg_paths):
                convert_npy_to_jpg(jpg_path)
                text = dataset.descriptions[jpg_path]
                texts.append(text)
            texts = clean_repeated_parts(texts)     
            
            if len(texts) != 5 or len(jpg_paths) != 5:
                continue
                
            jpg_paths = [os.path.join(data_dir, 'video_frames_sampled_jpg', f'{jpg_path}.jpg') for jpg_path in jpg_paths]
            
            row_data = {
                'img_seq_id':f'{split}_{item}',
                'img_paths': jpg_paths,
                'full_text': ' '.join(texts),
                'split': split
            }
            
            for i in range(5):
                row_data[f'link{i}'] = jpg_paths[i] if i < len(jpg_paths) else ''
                row_data[f'text{i}'] = texts[i] if i < len(texts) else ''
                
            all_data_flintstones.append(row_data)

    return pd.DataFrame(all_data_flintstones)

def main():
    parser = argparse.ArgumentParser(description="Process flintstones data")
    parser.add_argument("--data_dir", type=str, default="data/temporal_data/flintstones", help="Directory for data storage")
    parser.add_argument("--output_fn", type=str, default="filtered_flintstones_events.pkl", help="Output file name")
    parser.add_argument("--img_seq_only_criteria", type=str, default='link0', help="How to determine distinguished image sequence?")
    parser.add_argument("--select_shortest", type=bool, default=True, help="Select shortest story if multiple are available")
    parser.add_argument("--clip_sim_threshold", type=float, default=0, help="Threshold for clip similarity between image and text")
    parser.add_argument("--use_bertscore", type=bool, default=True, help="Create use bertscore for similarity calculation")
    parser.add_argument("--use_clipsim", type=bool, default=True, help="Use CLIP for similarity calculation")
    parser.add_argument("--filter_data", type=bool, default=False, help="Start with filtering data")
    parser.add_argument("--filtered_data_fn", type=str, default='data/temporal_data/flintstones/filtered_flintstones.csv', help="Filtered data file before extracting events")
    parser.add_argument("--text_sim_threshold", type=float, default=0.93, help="Similarity threshold for text filtering")
    
    parser.add_argument("--img_sim_threshold", type=float, default=0.90, help="Similarity threshold for image filtering")
    parser.add_argument("--text_thres_list", type=list, default=[0.96, 0.96, 0.95, 0.91, 0.91, 0.91], help="Similarity thresholds for bertscore's precsion, recall and f1")
    
    
    args = parser.parse_args()

    if args.filter_data:
    
        df_flintstones = load_flintstones_data(args.data_dir)
        
        filtered_flintstones = filter_data(df_flintstones, args.img_sim_threshold, args.img_seq_only_criteria, args.select_shortest, args.clip_sim_threshold, args.use_bertscore, args.use_clipsim, args.text_sim_threshold, args.text_thres_list)

        print(args.img_sim_threshold)
        print(args.text_thres_list)

        print(f"Original flintstones data: {len(df_flintstones)}")
        print(f"Filtered flintstones data: {len(filtered_flintstones)}")
        filtered_flintstones.to_csv(args.filtered_data_fn, index=False)
        combine_images(filtered_flintstones, 'flintstones', args.data_dir)
        
    else:
        filtered_flintstones = pd.read_csv(args.filtered_data_fn)  
        print(filtered_flintstones.shape)
        combine_images(filtered_flintstones, 'flintstones', args.data_dir)  
    sys.exit(0)

    
    print("Extract events ...")

    df_with_events = extract_events_in_batch(filtered_flintstones, 'flintstones', chunk_size=500)
    
    output_path = os.path.join(args.data_dir, args.output_fn)
    df_with_events.to_pickle(output_path)
    print(f"Filtered flintstones data with events saved to: {output_path}")
    
    
    print("Create questions ...")
    df_with_questions = create_questions(df_with_events)
    df_with_questions.to_pickle(os.path.join(args.data_dir, args.output_fn.replace('events', 'questions')))
    print(f"Filtered flintstones data saved to: {output_path}")
    
    
if __name__ == "__main__":
    main()