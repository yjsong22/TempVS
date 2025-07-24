import os
import json
import pandas as pd
import argparse
from data_filters import *
from extract_events import *
from PIL import Image, UnidentifiedImageError
import sys

def load_vist_data(args):
    train_data = json.load(open(os.path.join(args.sis_json_dir, 'train.story-in-sequence.json')))
    val_data = json.load(open(os.path.join(args.sis_json_dir, 'val.story-in-sequence.json')))
    test_data = json.load(open(os.path.join(args.sis_json_dir, 'test.story-in-sequence.json')))
    
    return train_data, val_data, test_data

def process_vist_data(train_data, val_data, test_data):
    splits = ["train", "val", "test"]
    whole_album = {}
    
    for i, split in enumerate(splits):
        album_mapping = {}
        count = 0
        for annot_new in [train_data, val_data, test_data][i]["annotations"]:
            annot = annot_new[0]
            assert len(annot_new) == 1
            story_id = annot['story_id']
                
            if story_id not in album_mapping:
                album_mapping[story_id] = {
                    "flickr_ids": [annot['photo_flickr_id']],
                    "sis": [annot['original_text']],
                    "length": 1,
                    "split": split,
                    "story_id": story_id,
                    "album_id": annot['album_id']
                }
            else:
                album_mapping[story_id]["flickr_ids"].append(annot['photo_flickr_id'])
                album_mapping[story_id]["sis"].append(annot['original_text'])
                album_mapping[story_id]["length"] += 1
        whole_album.update(album_mapping)
    
    return whole_album

def create_dataframe(whole_album, img_dir):
    df_whole_vist = pd.DataFrame.from_dict(whole_album, orient='index')
    
    def get_image_path(row, index):
        flickr_id = row['flickr_ids'][index]
        split = row['split']
        return f'{img_dir}/{split}/{flickr_id}.jpg'

    for i in range(5):
        df_whole_vist[f'link{i}'] = df_whole_vist.apply(lambda row: get_image_path(row, i), axis=1)
        df_whole_vist[f'text{i}'] = df_whole_vist.apply(lambda row: row['sis'][i] if i < len(row['sis']) else None, axis=1)

    
    df_whole_vist.dropna(subset=[f'link{i}' for i in range(5)] + [f'text{i}' for i in range(5)], inplace=True)

    def img_paths_exist_and_not_corrupted(row):
        for i in range(5):
            path = row.get(f'link{i}')
            
            # Check if path exists
            if path and not os.path.exists(path):
                return False
            
            # Try opening the image using Pillow
            try:
                with Image.open(path) as img:
                    # Check if the image is corrupted or unreadable
                    img.verify()  # This will raise an error if the image is not valid
            except (UnidentifiedImageError, OSError):
                return False
            
        return True

    df_whole_vist = df_whole_vist[df_whole_vist.apply(img_paths_exist_and_not_corrupted, axis=1)]

    df_whole_vist['full_text'] = df_whole_vist['sis'].apply(' '.join)

    print(f"Number of image sequences (without filtering): {len(df_whole_vist)}")
    
    return df_whole_vist


def main():
    parser = argparse.ArgumentParser(description="Process VIST data")
    parser.add_argument("--data_dir", type=str, default="data/temporal_data/vist", help="Directory for VIST data storage")
    parser.add_argument("--sis_json_dir", type=str, default="data/temporal_data/vist/annotations/sis", help="Directory for sis text storage")
    parser.add_argument("--img_dir", type=str, default="data/temporal_data/vist/images", help="Directory for image storage")
    parser.add_argument("--output_fn", type=str, default="filtered_vist_events.pkl", help="Output file name")
    parser.add_argument("--img_seq_only_criteria", type=str, default='album_id', help="How to determine distinguished image sequence?")
    parser.add_argument("--select_shortest", type=bool, default=False, help="Select shortest story if multiple are available")
    parser.add_argument("--clip_sim_threshold", type=float, default=0, help="Threshold for clip similarity between image and text")
    parser.add_argument("--use_bertscore", type=bool, default=True, help="Create use bertscore for similarity calculation")
    parser.add_argument("--use_clipsim", type=bool, default=True, help="Use CLIP for similarity calculation")
    parser.add_argument("--text_sim_threshold", type=float, default=0.96, help="Similarity threshold for text filtering")
    parser.add_argument("--img_sim_threshold", type=float, default=0.82, help="Similarity threshold for image filtering")
    parser.add_argument("--text_thres_list", type=list, default=[0.95, 0.95, 0.93, 0.90, 0.90, 0.90], help="Similarity thresholds for bertscore's precsion, recall and f1")
    parser.add_argument("--filter_data", type=bool, default=False, help="Start with filtering data")
    parser.add_argument("--filtered_data_fn", type=str, default='data/temporal_data/vist/filtered_vist_unextracted.csv', help="Filtered data file before extracting events")

    args = parser.parse_args()

    if args.filter_data:
        train_data, val_data, test_data = load_vist_data(args)
        whole_album = process_vist_data(train_data, val_data, test_data)
        df_whole_vist = create_dataframe(whole_album, args.img_dir)
        
        filtered_vist = filter_data(df_whole_vist, args.img_sim_threshold, args.img_seq_only_criteria, args.select_shortest, args.clip_sim_threshold, 
                                    args.use_bertscore, args.use_clipsim, args.text_sim_threshold, args.text_thres_list, check_person=True)

        print(args.img_sim_threshold)
        print(args.text_thres_list)    
        print(f"Original VIST data: {len(df_whole_vist)}")
        print(f"Filtered VIST data: {len(filtered_vist)}")
        
        filtered_vist.to_csv(args.filtered_data_fn, index=False)
        combine_images(filtered_vist, 'vist', args.data_dir)
    else:
        
        filtered_vist = pd.read_csv(args.filtered_data_fn)

    print(f"Filtered VIST data loaded from: {args.filtered_data_fn}")
    print(filtered_vist.shape)    


    print("Extract events ...")
    df_with_events = extract_events_in_batch(filtered_vist, 'vist', chunk_size=500)
    
    output_path = os.path.join(args.data_dir, args.output_fn)
    df_with_events.to_pickle(output_path)
    print(f"Filtered vist data with events saved to: {output_path}")
    

    

if __name__ == "__main__":
    main()