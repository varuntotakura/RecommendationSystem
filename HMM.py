import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from DatasetsPreprocess.HMM_Preprocess import HMM_Dataset, HMDataset
from Models.HMM_Model import HMModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def adjust_lr(optimizer, epoch):
    if epoch < 1:
        lr = 5e-5
    elif epoch < 6:
        lr = 1e-3
    elif epoch < 9:
        lr = 1e-4
    else:
        lr = 1e-5

    for p in optimizer.param_groups:
        p['lr'] = lr
    return lr
    
def get_optimizer(net):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=5e-5, betas=(0.9, 0.997),
                                 eps=1e-07)
    return optimizer

def calc_map(topk_preds, target_array, k=12):
    metric = []
    tp, fp = 0, 0
    
    for pred in topk_preds:
        if target_array[pred]:
            tp += 1
            metric.append(tp/(tp + fp))
        else:
            fp += 1
            
    return np.sum(metric) / min(k, target_array.sum())

def read_data(data):
    return tuple(d.cuda() for d in data[:-1]), data[-1].cuda()


def validate(model, val_loader, k=12):
    model.eval()
    
    tbar = tqdm(val_loader, file=sys.stdout)
    
    maps = []
    
    with torch.no_grad():
        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            logits = model(inputs)

            _, indices = torch.topk(logits, k, dim=1)

            indices = indices.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            
            for i in range(indices.shape[0]):
                maps.append(calc_map(indices[i], target[i]))
        
    return np.mean(maps)

def dice_loss(y_pred, y_true):
    y_pred = y_pred.sigmoid()
    intersect = (y_true*y_pred).sum(axis=1)
    
    return 1 - (intersect/(intersect + y_true.sum(axis=1) + y_pred.sum(axis=1))).mean()


def train(model, train_loader, val_loader, epochs):
    SEED = 0
    np.random.seed(SEED)
    
    optimizer = get_optimizer(model)
    scaler = torch.cuda.amp.GradScaler()
    
    criterion = torch.nn.functional.cross_entropy
    
    for e in range(epochs):
        model.train()
        tbar = tqdm(train_loader, file=sys.stdout)
        
        lr = adjust_lr(optimizer, e)
        
        loss_list = []

        for _, data in enumerate(tbar):
            inputs, target = read_data(data)

            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                logits = model(inputs)
                loss = criterion(logits, target.long())
            #loss.backward()
            scaler.scale(loss).backward()
            #optimizer.step()
            scaler.step(optimizer)
            scaler.update()
            
            loss_list.append(loss.detach().cpu().item())
            
            avg_loss = np.round(100*np.mean(loss_list), 4)

            tbar.set_description(f"Epoch {e+1} Loss: {avg_loss} lr: {lr}")
            
    return model

def create_test_dataset(df, result_df):
    WEEK_HIST_MAX = 5
    week = -1
    result_df["week"] = week
    hist_df = df[(df["week"] > week) & (df["week"] <= week + WEEK_HIST_MAX)]
    hist_df = hist_df.groupby("customer_id").agg({"article_id": list, "week": list}).reset_index()
    hist_df.rename(columns={"week": 'week_history'}, inplace=True)
    return result_df.merge(hist_df, on="customer_id", how="left")

def inference(le_article, model, loader, k=12):
    model.eval()
    tbar = tqdm(loader, file=sys.stdout)
    preds = []
    with torch.no_grad():
        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)
            logits = model(inputs)
            _, indices = torch.topk(logits, k, dim=1)
            indices = indices.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            for i in range(indices.shape[0]):
                preds.append(" ".join(list(le_article.inverse_transform(indices[i]))))
        
    
    return preds

def main():
    SEQ_LEN = 16
    BS = 256
    NW = 8
    MODEL_NAME = "HMM_Model"

    data, le_article, train_transactions, val_transactions, n_classes =  HMM_Dataset()
    val_dataset = HMDataset(val_transactions, SEQ_LEN)
    val_loader = DataLoader(val_dataset, batch_size=BS, shuffle=False, num_workers=NW,
                            pin_memory=False, drop_last=False)
    train_dataset = HMDataset(train_transactions, SEQ_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True, num_workers=NW,
                            pin_memory=False, drop_last=True)

    model = HMModel((len(le_article.classes_), 512), n_classes)
    model = model.to(device)
    model = train(model, train_loader, val_loader, epochs=10)

    result_df = pd.read_csv('../input/h-and-m-personalized-fashion-recommendations/sample_submission.csv').drop("prediction", axis=1)
    result_df = create_test_dataset(data, result_df)
    test_ds = HMDataset(result_df, SEQ_LEN, is_test=True)
    test_loader = DataLoader(test_ds, batch_size=BS, shuffle=False, num_workers=NW,
                            pin_memory=False, drop_last=False)
    result_df["prediction"] = inference(le_article, model, test_loader)
    result_df.to_csv("submission.csv", index=False, columns=["customer_id", "prediction"])
    print(result_df)

main()