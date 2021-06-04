import math

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score
from tqdm import tqdm


def get_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred, average='macro'),
        'f1-score': f1_score(y_true, y_pred, average='macro')
    }


def train(model, loader, optimizer, criterion, epochs=100, use_cuda=True):
    for epoch in range(epochs):
        with tqdm(loader) as pbar:
            epoch_desc = 'Epoch {{0: <{0}d}}'.format(1 + int(math.log10(epochs))).format(epoch + 1)
            pbar.set_description(epoch_desc)
            for inputs, targets in pbar:
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                preds = model(inputs)
                loss = criterion(preds, targets.long())

                if use_cuda:
                    targets = targets.cpu()
                    
                y_true = targets.numpy()
                y_pred = np.argmax(preds.detach().cpu().numpy(), axis=1)

                acc = accuracy_score(y_true, y_pred)
                if acc > 0:
                    print(acc)
                    
                pbar.set_postfix(loss='{0:.5f}'.format(loss), accuracy='{0:.03f}'.format(acc))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


def test(model, loader, use_cuda=True):
    model.eval()
    print('Eval model...')
    y_true = []
    y_pred = []
    for inputs, targets in tqdm(loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        
        preds = model(inputs)

        if use_cuda:
            targets = targets.cpu()
            
        y_true.extend(targets.numpy())
        y_pred.extend(np.argmax(preds.detach().cpu().numpy(), axis=1))

    return get_metrics(y_true, y_pred)
    