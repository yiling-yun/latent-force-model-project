import os
import torch
import pandas as pd
from model import LSTM, SocialGNN
from data import CharadeV2, split_train_val
from utils import set_seed
from main_cls import INPUT_MODE_CONFIG, train


if __name__ == '__main__':
    set_seed(1234)
    input_pkl = True
    norm = True
    use_relative = False
    one_hot = False
    input_mode = "coreAndForce" #"allForce" #core 
    reverse = False  # YY
    input_format = "excel" if not input_pkl else "pkl"
    infer_input_filename = '_shuffledInfer' if input_mode == "random" else ""
    label_filename = '' if not one_hot else '_onehot'

    try:
        input_mode_filename, input_size = INPUT_MODE_CONFIG[input_mode]
    except KeyError:
        raise ValueError(f"Unknown input_mode: {input_mode}")

    n_epochs = 200
    lr_search = [1e-2, 3e-3]
    n_layers_search = [2]  # [2, 3, 4]
    hidden_dim_search = [64]  # [64, 128, 256] #[512]
    batch_size_search = [1] # [1, 2, 4, 8, 16, 32]
    model_name = 'lstm'  # 'socialgnn' # 'lstm'
    apply_sigmoid = False
    checkpoint_path = 'checkpoints'
    os.makedirs(checkpoint_path, exist_ok=True)

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    # dataset
    excel_name = 'data/charade_traj_force_summary_normalized.pkl' if input_pkl else 'data/charades_traj_summary_selected1133.xlsx'
    video_file = 'data/charades_traj_all_without_inference.csv'
    label_distribution_file = 'data/one_hot.csv' if one_hot else 'data/732subj_response_distributions.csv'

    train_videos, val_videos = split_train_val(video_file)

    print(f"Split into {len(train_videos)} train and {len(val_videos)} val")

    train_dataset = CharadeV2(excel_name, train_videos, label_distribution_file, input_mode=input_mode, norm=norm, use_relative=use_relative, input_format = input_format)
    val_dataset = CharadeV2(excel_name, val_videos, label_distribution_file, input_mode=input_mode, norm=norm, use_relative=use_relative, input_format = input_format)
    num_classes = len(train_dataset.categories)

    print('Training...')
    total_max_val_acc = 0
    total_min_val_loss = 100

    results = []
    for batch_size in batch_size_search:
        for hidden_dim in hidden_dim_search:
            for n_layers in n_layers_search:
                for lr in lr_search:
                    print(f"================== Currently running input_mode={input_mode} batch_size={batch_size} lr={lr}, # layers={n_layers}, hidden_dim={hidden_dim} ============================")
                    filename = f"{model_name}{input_mode_filename}{infer_input_filename}{label_filename}_batch{batch_size}_lr{lr}_hd{hidden_dim}_nl{n_layers}"
                    model_filename = f"{model_name}{input_mode_filename}{label_filename}_batch{batch_size}_lr{lr}_hd{hidden_dim}_nl{n_layers}"  # reversed animations use the same model
                    if model_name == 'lstm':
                        model = LSTM(input_size=input_size, output_size=num_classes, hidden_dim=hidden_dim, n_layers=n_layers,
                                        drop_prob=0.5, apply_sigmoid=apply_sigmoid)
                    else:
                        model = SocialGNN(input_size=10, output_size=num_classes, spatial_dim=32, hidden_dim=hidden_dim,
                                        n_layers=n_layers, drop_prob=0.5, apply_sigmoid=apply_sigmoid)
                    model = model.to(device)
                    model_save_path = os.path.join(checkpoint_path, f'{model_filename}_best.pt')

                    min_val_loss, max_val_acc = train(model, train_dataset, val_dataset, model_save_path, n_epochs,
                                                    lr, batch_size, device, filename)
                    if min_val_loss <= total_min_val_loss:
                        total_min_val_loss = min_val_loss
                        min_loss_modelname = f"batch{batch_size}_lr{lr}_hd{hidden_dim}_nl{n_layers}"

                    if max_val_acc >= total_max_val_acc:
                        total_max_val_acc = max_val_acc
                        max_acc_modelname = f"batch{batch_size}_lr{lr}_hd{hidden_dim}_nl{n_layers}"

                    # ---- save metrics for this config ----
                    results.append({
                        "batch_size": batch_size,
                        "hidden_dim": hidden_dim,
                        "n_layers": n_layers,
                        "lr": lr,
                        "min_val_loss": min_val_loss,
                        "max_val_acc": max_val_acc,
                        "model_filename": model_filename,
                        "log_filename": filename,
                    })

    # only save if we are doing hyperparameter grid search
    if len(results) > 1:
        print(f"Total min loss model parameters: {min_loss_modelname}")
        print(f"Total max acc model parameters: {max_acc_modelname}")
        results_df = pd.DataFrame(results)
        results_csv_path = os.path.join(checkpoint_path, f"swift_gridsearch_results_{input_mode}.csv")
        results_df.to_csv(results_csv_path, index=False)

        print(f"Saved results to: {results_csv_path}")
    else:
        print(results)
