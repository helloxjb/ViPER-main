import logging
import wandb
import time
import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from collections import OrderedDict
from scipy.linalg import sqrtm
from sklearn.preprocessing import StandardScaler
_logger = logging.getLogger("train")


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# 带温度的MSP
def compute_msp_with_temperature(cfg, logits):
    temperature = cfg["TRAINING"]["temperature"]
    scaled_logits = logits / temperature
    softmax_probs = F.softmax(scaled_logits, dim=1)
    msp = softmax_probs.max(dim=1)[0]
    return msp

def train(
    cfg, model, dataloader, criterion, optimizer, log_interval: int, device: str
) -> dict:
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    acc_m = AverageMeter()
    losses_m = AverageMeter()
    num_classes = cfg["DATASET"]["num_classes"]
    ratio = cfg["TRAINING"]["split_ratio"]
    
    end = time.time()

    model.train()
    optimizer.zero_grad()
    epoch_num = 0

    for idx, (inputs, targets) in enumerate(dataloader):     
        
        data_time_m.update(time.time() - end)
        inputs,targets=inputs.to(device),targets.to(device)
        
        # predict
        outputs = model(inputs)
        loss = criterion(outputs,targets)
        # loss.backward()
        
        # shuffle
        open_outputs = model(inputs, shuffle=True) # shuffle
        batch_size = inputs.size(0)
        open_targets = torch.full((batch_size, num_classes),
                                  1.0/num_classes,
                                  device=device)
        open_loss = criterion(open_outputs, open_targets)
        
        total_loss = loss+open_loss*cfg["TRAINING"]["alpha"]
        total_loss.backward()

        # loss update
        optimizer.step()
        optimizer.zero_grad()
        losses_m.update(loss.item())
        
        epoch_num += 1
           
        # accuracy
        stacked_outputs = torch.stack([outputs], dim=0)
        outputs = torch.mean(stacked_outputs, dim=0).squeeze(0)
        preds = outputs.argmax(dim=1)
        acc_m.update(
            targets.eq(preds).sum().item() / targets.size(0), n=targets.size(0)
        )

        batch_time_m.update(time.time() - end)

        if idx % log_interval == 0:
            _logger.info(
                "TRAIN [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) "
                "Acc: {acc.avg:.3%} "
                "LR: {lr:.3e} "
                "Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) "
                "Data: {data_time.val:.3f} ({data_time.avg:.3f})".format(
                    idx + 1,
                    len(dataloader),
                    loss=losses_m,
                    acc=acc_m,
                    lr=optimizer.param_groups[0]["lr"],
                    batch_time=batch_time_m,
                    rate=inputs.size(0) / batch_time_m.val,
                    rate_avg=inputs.size(0) / batch_time_m.avg,
                    data_time=data_time_m,
                )
            )

        end = time.time()
    
    return OrderedDict([("acc", acc_m.avg), ("loss", losses_m.avg)])


def test(
    cfg, model, testloader, openloader, criterion, log_interval: int, device: str, step
) -> dict:
    correct = 0
    total = 0
    total_loss = 0

    total_logits = []
    total_targets = []
    in_confidences = []
    out_confidences = []
    num_classes = cfg["DATASET"]["num_classes"]

    model.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # predict
            outputs = model(inputs)
            # msp
            in_msp = compute_msp_with_temperature(cfg, outputs)
            in_confidence = in_msp.cpu().numpy()
            in_confidences.extend(list(in_confidence))
            # loss
            loss = criterion(outputs, targets)
            total_targets.extend(list(targets.cpu().numpy()))
            total_logits.append(outputs)

            # total loss and acc
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += targets.eq(preds).sum().item()
            total += targets.size(0)

            if idx % log_interval == 0:
                _logger.info(
                    "TEST [%d/%d]: Loss: %.3f | Acc: %.3f%% [%d/%d]"
                    % (
                        idx + 1,
                        len(testloader),
                        total_loss / (idx + 1),
                        100.0 * correct / total,
                        correct,
                        total,
                    )
                )

        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        joint_logits_tensor = torch.tensor(joint_logits)
        softmax_close = torch.softmax(joint_logits_tensor, dim=1)
        pred_close = torch.argmax(softmax_close, dim=1)
        prob_close = softmax_close.max(1)[0]
        pred_close = pred_close.cpu().numpy()
        labels_close = total_targets

        # openset
        # reset total
        total_targets = []
        total_logits = []

        for idx, (inputs, targets) in enumerate(openloader):
            inputs, targets = inputs.to(device), targets.to(device)
            ood_label = torch.zeros_like(targets) - 1

            outputs = model(inputs)
            # msp
            out_msp = compute_msp_with_temperature(cfg, outputs)
            out_confidence = out_msp.cpu().numpy()
            out_confidences.extend(list(out_confidence))
            total_targets.extend(list(targets.cpu().numpy()))
            total_logits.append(outputs)

            if idx % log_interval == 0:
                _logger.info("OSR TEST [%d/%d]" % (idx + 1, len(openloader)))

        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        joint_logits_tensor = torch.tensor(joint_logits)
        softmax_open = torch.softmax(joint_logits_tensor, dim=1)
        pred_open = torch.argmax(softmax_open, dim=1)
        prob_open = softmax_open.max(1)[0]
        pred_open = pred_open.cpu().numpy()
        labels_open = total_targets
        labels_open = np.zeros_like(total_targets) - 1

        # F1 score
        total_pred_label = np.concatenate([pred_close, pred_open], axis=0)
        total_label = np.concatenate([labels_close, labels_open], axis=0)

        # mark test images in open_set
        open_labels = np.zeros(len(labels_close) + len(labels_open))
        for t in range(len(labels_close)):
            open_labels[t] = 1

        # AUROC score
        total_prob = np.concatenate([in_confidences, out_confidences], axis=0)
        prob = total_prob
        prob = prob.reshape(-1, 1)
        # print(open_labels.size, prob.size)
        fpr, tpr, thresholds = roc_curve(open_labels, prob)
        auroc = auc(fpr, tpr)

        thresh_idx = np.abs(np.array(tpr) - 0.95).argmin()
        threshold = thresholds[thresh_idx]
        fpr_tpr95 = fpr[thresh_idx]
        det_acc = 0.5 * (tpr + 1.0 - fpr).max()

        open_pred = (total_prob > threshold).astype(np.float32)
        micro_f1 = f1_score(
            total_label, ((total_pred_label + 1) * open_pred) - 1, average="micro"
        )
        macro_f1 = f1_score(
            total_label, ((total_pred_label + 1) * open_pred) - 1, average="macro"
        )

        precision, recall, _ = precision_recall_curve(open_labels, prob)
        aupr_in = auc(recall, precision)
        precision, recall, _ = precision_recall_curve(
            np.bitwise_not((open_labels).astype(bool)), -prob
        )
        aupr_out = auc(recall, precision)

        # OSCR
        in_confidences = np.array(in_confidences)
        out_confidences = np.array(out_confidences)
        total_confidences = np.concatenate([in_confidences, out_confidences])
        pred_close = np.array(pred_close)
        labels_close = np.array(labels_close)
        # 降序生成阈值
        thresholds = np.sort(total_confidences)[::-1]

        ccr_list = []
        fpr_list = []

        for t in thresholds:
            # 已知类中预测正确且置信度>t
            correct_mask = pred_close == labels_close
            above_threshold = in_confidences > t
            ccr = np.sum(correct_mask & above_threshold) / len(labels_close)
            # 未知类中置信度>t
            fpr = np.sum(out_confidences > t) / len(out_confidences)

            ccr_list.append(ccr)
            fpr_list.append(fpr)

        # OSCR AUC
        oscr_auc = auc(fpr_list, ccr_list)

        _logger.info(
            "AUROC: {:.5f}, AUPR_IN: {:.5f}, AUPR_OUT: {:.5f}, Macro F1-score: {:.5f}, Micro F1-score: {:.5f}, Det_ACC: {:.5f}, FPR@TPR95: {:.5f}, OSCR_AUC: {:.5f}".format(
                auroc,
                aupr_in,
                aupr_out,
                macro_f1,
                micro_f1,
                det_acc,
                fpr_tpr95,
                oscr_auc,
            )
        )
    
        

    return OrderedDict(
        [
            ("acc", correct / total),
            ("loss", total_loss / len(testloader)),
            ("AUROC", auroc),
            ("AUPR_IN", aupr_in),
            ("AUPR_OUT", aupr_out),
            ("Macro F1-score", macro_f1),
            ("Micro F1-score", micro_f1),
            ("Det_ACC", det_acc),
            ("FPR@TPR95", fpr_tpr95),
            ("OSCR AUC", oscr_auc),
        ]
    )
    


def fit(
    cfg,
    model,
    trainloader,
    testloader,
    openloader,
    criterion,
    optimizer,
    scheduler,
    epochs: int,
    savedir: str,
    log_interval: int,
    device: str,
    use_wandb: bool,
) -> None:

    best_acc = 0
    step = 0
    for epoch in range(epochs):
        _logger.info(f"\nEpoch: {epoch+1}/{epochs}")
        train_metrics = train(
            cfg, model, trainloader, criterion, optimizer, log_interval, device
        )
        eval_metrics = test(
            cfg, model, testloader, openloader, criterion, log_interval, device, step
        )

        if use_wandb:
            # wandb
            metrics = OrderedDict(lr=optimizer.param_groups[0]["lr"])
            metrics.update([("train_" + k, v)
                           for k, v in train_metrics.items()])
            metrics.update([("eval_" + k, v) for k, v in eval_metrics.items()])
            wandb.log(metrics, step=step)

        step += 1

        # step scheduler
        if scheduler:
            scheduler.step()

        # checkpoint
        if best_acc < eval_metrics["acc"]:
            # save results
            state = {"best_epoch": epoch, "best_acc": eval_metrics["acc"]}
            json.dump(
                state, open(os.path.join(savedir, f"best_results.json"), "w"), indent=4
            )

            # save model
            torch.save(model.state_dict(), os.path.join(
                savedir, f"best_model.pt"))

            _logger.info(
                "Best Accuracy {0:.3%} to {1:.3%}".format(
                    best_acc, eval_metrics["acc"])
            )

            best_acc = eval_metrics["acc"]

    _logger.info(
        "Best Metric: {0:.3%} (epoch {1:})".format(
            state["best_acc"], state["best_epoch"]
        )
    )
