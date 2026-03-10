import pandas as pd
from utils import set_seed
from data import read_and_concat


def split_and_save(filename, video_names):
    ids = [v.split('_')[0] for v in video_names]
    labels = [v.split('_')[1] for v in video_names]

    df = pd.DataFrame({'ID': ids, 'label': labels})
    df.to_csv(filename, index=False)


if __name__ == '__main__':
    set_seed(1234)
    full_traj_file = '../data/charades_traj_summary_selected1133.xlsx'  #with short videos removed #'data/charades_traj_summary_full.xlsx'
    old_test_file = '../data/charades_traj_summary_full_test.xlsx'

    infer_files = ['../data/charades_participant_diffStim.xlsx', '../data/charades_participant.xlsx']

    all_data_file = '../data/charades_traj_all_without_inference.csv'

    infer_videos = []
    for infer_f in infer_files:
        infer_videos.extend(read_and_concat(infer_f))

    infer_videos = list(set(infer_videos))
    print(f"Number of unique inference videos {len(infer_videos)}")

    test_videos = read_and_concat(old_test_file)
    print(f"Number of test videos: {len(test_videos)}")

    all_videos = read_and_concat(full_traj_file)
    print(f"Number of all videos: {len(all_videos)}")

    all_videos = list(set(all_videos + test_videos))
    print(f"Number of all videos after combined: {len(all_videos)}")

    for infer_video in infer_videos:
        if infer_video not in all_videos:
            print(f"No trajectory data for inference video {infer_video}")

    all_videos = [t for t in all_videos if t not in infer_videos]

    print('--------- After filtering -------------')
    print(f"Number of all videos excluding infernece: {len(all_videos)}")

    split_and_save(all_data_file, all_videos)
