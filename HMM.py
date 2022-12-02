import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Importing the custom files and corresponding classess and functions
from DatasetsPreprocess.HMM_Preprocess import HMM_Dataset, HMDataset
from Models.HMM_Model import HMModel

# Setting the device as CUDA if CUDA is available, if not to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def learning_rate(optimizer, epoch):
    '''
        Variating the learing rate of the mdoel so that it could learn the training data without 
    '''
    if epoch < 1:
        lr = 5e-5
    elif epoch < 6:
        lr = 1e-3
    elif epoch < 9:
        lr = 1e-4
    else:
        lr = 1e-5
    for param in optimizer.param_groups:
        param['lr'] = lr
    return lr
    
def get_optimizer(net):
    '''
        Initializing the optimizer and starting learning rate
    '''
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-5, weight_decay=0.9, eps=1e-07)

    return optimizer

def mapping(topk_preds, target_array, k=12):
    '''
        Getting top 12 results from teh target array using TopK Algorithm
    '''
    metric = []
    true_positive, false_positive = 0, 0
    for pred in topk_preds:
        if target_array[pred]:
            true_positive += 1
            metric.append(true_positive/(true_positive + false_positive))
        else:
            false_positive += 1
    return np.sum(metric) / min(k, target_array.sum())

def read_data(data):
    '''
        Read the data into device
    '''
    return tuple(d.to(device) for d in data[:-1]), data[-1].to(device)

def train(model, train_loader, n_epochs):
    '''
        Training the model
    '''
    optimizer = get_optimizer(model)
    criterion = torch.nn.functional.cross_entropy
    for epoch in range(n_epochs):
        model.train()
        lr = learning_rate(optimizer, epoch)
        loss_list = []
        for _, data in enumerate(tqdm(train_loader)):
            inputs, target = read_data(data)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(inputs)
                loss = criterion(logits, target.long())
            loss.backward()
            optimizer.step()
            loss_list.append(loss.detach().cpu().item())
            avg_loss = np.round(100*np.mean(loss_list), 4)
        print(f"Epoch {epoch+1} Loss: {avg_loss} lr: {lr}")
            
    return model, loss_list

def format_test_dataset(df, result_df):
    '''
        Format the dataset
    '''
    WEEK_HIST_MAX = 5
    week = -1
    result_df["week"] = week
    hist_df = df[(df["week"] > week) & (df["week"] <= week + WEEK_HIST_MAX)]
    hist_df = hist_df.groupby("customer_id").agg({"article_id": list, "week": list}).reset_index()
    hist_df.rename(columns={"week": 'week_history'}, inplace=True)
    return result_df.merge(hist_df, on="customer_id", how="left")

def label_decoder(label_encoder, model, loader, k=12):
    '''
        Decode the labels into its original formata and pick top 12 items using the TopK Algorithm
    '''
    model.eval()
    preds = []
    with torch.no_grad():
        for idx, data in enumerate(tqdm(loader)):
            inputs, target = read_data(data)
            logits = model(inputs)
            _, indices = torch.topk(logits, k, dim=1)
            indices = indices.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            for i in range(indices.shape[0]):
                str_arr = []
                for s in indices[i]:
                    str_arr.append(s)
                preds.append(label_encoder.inverse_transform(str_arr))
    return preds

def main():
    SEQ_LEN = 32
    BATCH_SIZE = 128
    NUM_WORKERS = 16

    data, label_encoder, train_transactions, val_transactions, n_classes =  HMM_Dataset()
    val_dataset = HMDataset(val_transactions, SEQ_LEN)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                            pin_memory=False, drop_last=False)
    train_dataset = HMDataset(train_transactions, SEQ_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                            pin_memory=False, drop_last=True)

    model = HMModel((len(label_encoder.classes_), 512), n_classes)
    model = model.to(device)
    model, loss_list = train(model, train_loader, n_epochs=10)
    torch.save(model, "./Checkpoints/HMM.pth")
    model = torch.load("./Checkpoints/HMM.pth")
    result_df = pd.read_csv('../input/h-and-m-personalized-fashion-recommendations/sample_submission.csv').drop("prediction", axis=1)
    result_df = format_test_dataset(data, result_df)
    test_ds = HMDataset(result_df, SEQ_LEN, is_test=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                            pin_memory=False, drop_last=False)
    result_df["prediction"] = label_decoder(label_encoder, model, test_loader)
    result_df.to_csv("./Results/HMM_Results.csv", index=False, columns=["customer_id", "prediction"])
    print(result_df.head())
    plt.plot(loss_list)
    plt.show()

main()