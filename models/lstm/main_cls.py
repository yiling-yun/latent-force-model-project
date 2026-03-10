import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from model import LSTM, SocialGNN
from data import CharadeV2, cls_collate_fn, BalancedBatchSampler
from torch.utils.data import DataLoader
from utils import set_seed, soft_ce_loss, plot_smoothed_loss, EarlyStopping


INPUT_MODE_CONFIG = {
    "core": ("_core", 9),
    "coreAndForce": ("_coreAndForce", 18),
    "allForce": ("_force", 9),
}


def val(model, dataset, device, batch_size=1):
    val_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # count = 0
    e_loss = 0
    model.eval()

    acc_list = []
    for data, id, videoLabel, label in val_dataloader:
        data, label = data.float().to(device), label.float().to(device)

        logits, anchor_hidden = model(data)
        loss = soft_ce_loss(logits, label)

        preds = torch.softmax(logits, dim=-1)  # predicted distribution
        pred_class = preds.argmax(dim=-1)  # most probable class
        true_class = label.argmax(dim=-1)  # most probable true class
        acc = (pred_class == true_class).float().cpu().numpy()
        acc_list.append(acc)

        # count += 1
        e_loss += loss.item()

    # e_loss /= float(count)
    e_loss /= float(len(val_dataloader))
    acc = np.concatenate(acc_list).mean()

    return e_loss, acc


def train(model, train_dataset, val_dataset, model_save_path, n_epochs, lr, batch_size, device, filename, log_dir='results'):
    os.makedirs(log_dir, exist_ok=True)

    batch_sampler = BalancedBatchSampler(train_dataset, batch_size)
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        collate_fn=cls_collate_fn
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.4 * n_epochs), gamma=0.1)
    early_stopping = EarlyStopping(patience=int(0.5 * n_epochs), verbose=False)
    #early_stopping = EarlyStopping(patience=int(20), verbose=False)

    min_loss = 100
    max_acc = 0
    e_loss_list = []
    val_e_loss_list = []
    val_acc_list = []
    pbar = tqdm(range(n_epochs), desc="Training", unit="epoch")
    for e in pbar:
        model.train()
        e_loss = 0

        for data, lengths, id, videoLabel, label in train_dataloader:
            data, label, lengths = data.float().to(device), label.float().to(device), lengths.to(device)

            logits, anchor_hidden = model(data, lengths=lengths) #YY: add lengths =
            loss = soft_ce_loss(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            e_loss += loss.item()

        e_loss /= len(train_dataloader)
        e_loss_list.append(e_loss)
        scheduler.step()

        val_e_loss, val_acc = val(model, val_dataset, device)
        val_e_loss_list.append(val_e_loss)
        val_acc_list.append(val_acc)

        pbar.set_postfix({
            "train_loss": f"{e_loss:.4f}",
            "val_loss": f"{val_e_loss:.4f}",
            "val_acc": f"{val_acc:.3f}"
        })

        if min_loss >= val_e_loss:
            min_loss = val_e_loss
            torch.save(model.state_dict(), model_save_path)

        if max_acc <= val_acc:
            max_acc = val_acc
        torch.save(model.state_dict(), model_save_path.replace('best', 'last'))

        filename_loss = f'{log_dir}/{filename}_soft_ce_loss.png'
        plot_smoothed_loss(e + 1, n_epochs, e_loss_list, val_e_loss_list, is_acc=False, filename=filename_loss)

        early_stopping(val_e_loss)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # saving the loss in each epoch
    result_df = pd.DataFrame({"Epoch": range(e + 1),
                              "TrainLoss": e_loss_list,
                              "ValLoss": val_e_loss_list,
                              "ValAcc": val_acc_list})
    result_df.to_csv(f"{log_dir}/{filename}_loss.csv")

    return min_loss, max_acc


def inference_similarity(model, dataset, video_id, device):
    model.eval()
    out = {}
    for idx in range(len(dataset)):
        data, id, videoLabel, label = dataset[idx]
        if id in video_id:
            data = torch.FloatTensor(data).unsqueeze(0).to(device)

            logits, hidden = model(data)  # logits shape: [1, C]

            probs = torch.softmax(logits, dim=-1)  # predicted distribution

            out[id] = {
                "feat": hidden.squeeze(0).detach().cpu().numpy(),
                "probs": probs.squeeze(0).detach().cpu().numpy(),  # distribution over classes
            }
    return out


if __name__ == '__main__':
    set_seed(1234)
    input_pkl = True  
    norm = True
    use_relative = False
    one_hot = False
    input_mode = "core" #"allForce" #core # coreAndForce 
    reverse = False  
    input_format = "excel" if not input_pkl else "pkl"
    infer_input_filename = '_shuffledInfer' if input_mode == "random" else ""
    label_filename = '' if not one_hot else '_onehot'
   
    try:
        input_mode_filename, input_size = INPUT_MODE_CONFIG[input_mode]
    except KeyError:
        raise ValueError(f"Unknown input_mode: {input_mode}")

    n_epochs = 200
    lr = 0.003   
    hidden_dim = 64 
    n_layers = 2 
    batch_size = 1
    model_name = 'lstm'  # 'socialgnn' # 'lstm'
    apply_sigmoid = False
    checkpoint_path = 'checkpoints'
    os.makedirs(checkpoint_path, exist_ok=True)

    print(
        f"================== Currently running input_mode={input_mode} batch_size={batch_size} lr={lr}, # layers={n_layers}, hidden_dim={hidden_dim} ============================")

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
    inference_file = ['data/charades_participant_diffStim.xlsx', 'data/charades_participant.xlsx']
    label_distribution_file = 'data/one_hot.csv' if one_hot else 'data/732subj_response_distributions.csv'

    train_dataset = CharadeV2(excel_name, video_file, label_distribution_file, input_mode=input_mode, norm=norm, use_relative=use_relative, input_format = input_format)
    num_classes = len(train_dataset.categories)

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

    train(model, train_dataset, train_dataset, model_save_path, n_epochs, lr, batch_size, device, filename)

    print('-' * 60)
    print('Inferencing...')
    model.load_state_dict(torch.load(model_save_path.replace('best', 'last')))
    model.eval()

    for infer_f in inference_file:
        feat_save_path = f"results/{filename}_{os.path.basename(infer_f).split('.')[0]}_soft_ce_loss_charades_feat.npy"
        infer_dataset = CharadeV2(excel_name, infer_f, input_mode=input_mode, norm=norm, use_relative=use_relative, input_format = input_format)  # 27 animations
        participant_videos = pd.read_excel(infer_f, engine="openpyxl")
        infer_video_id = list(participant_videos['ID'])
        feat_dict = inference_similarity(model, infer_dataset, infer_video_id, device)
        np.save(feat_save_path, feat_dict)
        print(f"Saved {feat_save_path}")
    print('Inference finished.')

