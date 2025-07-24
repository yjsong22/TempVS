from data_filters import *
from extract_events import *
import pandas as pd
import os
import json
import argparse
from datasets import load_dataset
import sys


def main():
    parser = argparse.ArgumentParser(description="Process vwp data")
    parser.add_argument("--data_dir", type=str, default="data/temporal_data/vwp", help="Directory for data storage")
    parser.add_argument("--output_fn", type=str, default="filtered_vwp_events.pkl", help="Output file name")
    parser.add_argument("--img_seq_only_criteria", type=str, default='link0', help="How to determine distinguished image sequence?")
    parser.add_argument("--select_shortest", type=bool, default=True, help="Select shortest story if multiple are available")
    parser.add_argument("--clip_sim_threshold", type=float, default=0, help="Threshold for clip similarity between image and text")
    parser.add_argument("--use_bertscore", type=bool, default=True, help="Create use bertscore for similarity calculation")
    parser.add_argument("--use_clipsim", type=bool, default=True, help="Use CLIP for similarity calculation")
    parser.add_argument("--text_sim_threshold", type=float, default=0.96, help="Similarity threshold for text filtering")
    parser.add_argument("--img_sim_threshold", type=float, default=0.95, help="Similarity threshold for image filtering")
    parser.add_argument("--text_thres_list", type=list, default=[0.97, 0.97, 0.95, 0.92, 0.92, 0.92], help="Similarity thresholds for bertscore's precsion, recall and f1")
    parser.add_argument("--filter_data", type=bool, default=True, help="Start with filtering data")
    parser.add_argument("--filtered_data_fn", type=str, default='data/temporal_data/vwp/filtered_vwp.csv', help="Filtered data file before extracting events")

    args = parser.parse_args()

    if args.filter_data:
    
        dataset = load_dataset("tonyhong/vwp")
        vwp_train = pd.DataFrame(dataset['train'])
        vwp_val = pd.DataFrame(dataset['val'])
        
        df_vwp_concat = pd.concat([vwp_train, vwp_val])
        df_vwp = df_vwp_concat.copy()
        
        for col in df_vwp_concat.columns:
            if col.startswith('link'):
                df_vwp[col] = df_vwp_concat[col].apply(lambda url: url.replace("https://datasets.d2.mpi-inf.mpg.de/xhong/VST", "data/temporal_data/vwp") if pd.notnull(url) else url)
                df_vwp['full_text'] = df_vwp_concat['story']
        
        filtered_vwp = filter_data(df_vwp,args.img_sim_threshold, args.img_seq_only_criteria, args.select_shortest, args.clip_sim_threshold, args.use_bertscore, args.use_clipsim, args.text_sim_threshold, args.text_thres_list)
        
        print(args.img_sim_threshold)
        print(args.text_thres_list)
        print(f"Original VWP data: {len(df_vwp)}")
        print(f"Filtered VWP data: {len(filtered_vwp)}")
        
        filtered_vwp.to_csv(args.filtered_data_fn, index=False)
        print(f"Filtered VWP data saved to: {args.filtered_data_fn}")
        combine_images(filtered_vwp, 'vwp', args.data_dir)
    
    else:
        filtered_vwp = pd.read_csv(args.filtered_data_fn)
    
    print("Extract events ...")

    df_with_events = extract_events_in_batch(filtered_vwp, 'vwp', chunk_size=500, time_limit=60)

    output_path = os.path.join(args.data_dir, args.output_fn)
    df_with_events.to_pickle(output_path)
    print(f"Filtered vwp data with events saved to: {output_path}")
    
    print("Create questions ...")
    df_with_questions = create_questions(df_with_events)
    df_with_questions.to_pickle(os.path.join(args.data_dir, args.output_fn.replace('events', 'questions')))
    
    
if __name__ == "__main__":
    main()