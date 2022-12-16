import os
from tqdm import tqdm
import torch
import numpy as np
import logging
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score
from judge import AverageMeter

from utils import f1_cal, save_checkpoint

def train_auxi(args, tune_domain_loader, DEVICE, model, optimizer, e, false_optimizer):
    model.train()
    train_loss = 0
    epoch_class_y_loss = 0
    total = 0

    for idx, domain_loader in enumerate(tune_domain_loader):
        for (x, y, d) in domain_loader:
            # if idx == 119:
            #     print("now")
            x, y, d = x.to(DEVICE), y.to(DEVICE), d.to(DEVICE)

            optimizer.zero_grad()
            false_optimizer.zero_grad()
            
            if x.shape[0] == 1:
                # there are cases that only one label is provided by one user
                x = torch.concat([x, x], dim=0)  # to make batch normalization layer can run 
                y = torch.concat([y, y], dim=0)
                d = torch.concat([d, d], dim=0)

            loss_origin, class_y_loss = model.loss_function(d, x, y)
            # if torch.any(torch.isnan(loss_origin)):
            #     print('now')
            
            loss_origin = loss_origin

            loss_false = model.loss_function_false(args, d, x, y)

            loss_origin.backward()
            optimizer.step()
            loss_false.backward()
            false_optimizer.step()

            train_loss += loss_origin
            epoch_class_y_loss += class_y_loss
            total += y.size(0)

        train_loss /= total
        epoch_class_y_loss /= total

    return train_loss, epoch_class_y_loss


def get_f1(source_loaders, DEVICE, model, classifier_fn, batch_size):
    model.eval()
    """
    compute the accuracy over the supervised training set or the testing set
    """

    f1_batch = AverageMeter('f1_eval', ':6.2f')
    
    with torch.no_grad():
        for source_loader in source_loaders:
            for (xs, ys, ds) in source_loader:

                xs, ys, ds = xs.to(DEVICE), ys.to(DEVICE), ds.to(DEVICE)

                # use classification function to compute all predictions for each batch
                _, pred_y, _, _ = classifier_fn(xs)

                _, pred = pred_y.topk(1, 1, True, True)

                f1 = f1_score(ys.cpu().numpy(), pred.cpu().numpy(), average='macro') * 100

                f1_batch.update(f1, xs.size(0))

    return f1_batch.avg
    

def get_accuracy(source_loaders, DEVICE, model, classifier_fn, batch_size):
    model.eval()
    """
    compute the accuracy over the supervised training set or the testing set
    """
    predictions_d, actuals_d, predictions_y, actuals_y = [], [], [], []
    predictions_d_false, predictions_y_false = [], []

    with torch.no_grad():
        for source_loader in source_loaders:
            for (xs, ys, ds) in source_loader:

                xs, ys, ds = xs.to(DEVICE), ys.to(DEVICE), ds.to(DEVICE)

                # use classification function to compute all predictions for each batch
                pred_d, pred_y, pred_d_false, pred_y_false = classifier_fn(xs)
                
                predictions_d.append(pred_d)
                predictions_d_false.append(pred_d_false)
                actuals_d.append(ds)

                predictions_y.append(pred_y)
                predictions_y_false.append(pred_y_false)
                actuals_y.append(ys)

        # compute the number of accurate predictions
        accurate_preds_d = 0
        accurate_preds_d_false = 0
        for pred, act in zip(predictions_d, actuals_d):
            _, num_pred = pred.max(1)
            v = torch.sum(num_pred == act)
            accurate_preds_d += v

        accuracy_d = (accurate_preds_d * 100.0) / (len(predictions_d) * batch_size)

        for pred, act in zip(predictions_d_false, actuals_d):
            _, num_pred = pred.max(1)
            v = torch.sum(num_pred == act)
            accurate_preds_d_false += v

        # calculate the accuracy between 0 and 1
        accuracy_d_false = (accurate_preds_d_false * 100.0) / (len(predictions_d_false) * batch_size)

        # compute the number of accurate predictions
        accurate_preds_y = 0
        accurate_preds_y_false = 0

        for pred, act in zip(predictions_y, actuals_y):
            _, num_pred = pred.max(1)
            v = torch.sum(num_pred == act)
            accurate_preds_y += v

        accuracy_y = (accurate_preds_y * 100.0) / (len(predictions_y) * batch_size)


        for pred, act in zip(predictions_y_false, actuals_y):
            _, num_pred = pred.max(1)
            v = torch.sum(num_pred == act)
            accurate_preds_y_false += v
        # calculate the accuracy between 0 and 1
        accuracy_y_false = (accurate_preds_y_false * 100.0) / (len(predictions_y_false) * batch_size)

        return accuracy_d, accuracy_y, accuracy_d_false, accuracy_y_false



def train_GILE(model, DEVICE, optimizer, tune_loader, val_loader, test_loader, args):

    writer_pos = './runs/' + args.store + '/' + args.name
    writer = SummaryWriter(writer_pos)
    logging.basicConfig(filename=os.path.join(writer.log_dir, 'training.log'), level=logging.DEBUG)


    logging.info(f"Start GILE trainign for {args.n_epoch} epochs.")

    best_f1, best_epoch = 0.0, 0

    false_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


    for e in tqdm(range(args.n_epoch)):
        
        # training 
        avg_epoch_loss, avg_epoch_class_y_loss = train_auxi(args, tune_loader, DEVICE, model, optimizer, e, false_optimizer)

        # evaluate
        # train_f1_d, train_f1_y, train_f1_d_false, train_f1_y_false = get_accuracy(tune_loader, DEVICE, model, model.classifier, args.batch_size)
        # val_f1_d, val_f1_y, val_f1_d_false, val_f1_y_false = get_accuracy([val_loader], DEVICE, model, model.classifier, args.batch_size)
        # test_f1_d, test_f1_y, test_f1_d_false, test_f1_y_false = get_accuracy([test_loader], DEVICE, model, model.classifier, args.batch_size)

        train_f1 = get_f1(tune_loader, DEVICE, model, model.classifier, args.batch_size)
        val_f1 = get_f1([val_loader], DEVICE, model, model.classifier, args.batch_size)
        test_f1 = get_f1([test_loader], DEVICE, model, model.classifier, args.batch_size)

        writer.add_scalar('loss', avg_epoch_loss, global_step=e)
        writer.add_scalar('loss_y', avg_epoch_class_y_loss, global_step=e)
        writer.add_scalar('train_f1', train_f1, global_step=e)
        writer.add_scalar('val_f1', val_f1, global_step=e)

        logging.debug('Epoch: [{}/{}], Avg loss: {:.2f}, y loss: {:.2f}%'.format(e + 1, args.n_epoch, avg_epoch_loss, avg_epoch_class_y_loss))
        # logging.debug('Epoch:[{}/{}], train f1 d:{:.2f} y:{:.2f} | d_false:{:.2f} y_false:{:.2f}'.format(e+1, args.n_epoch, train_f1_d, train_f1_y, train_f1_d_false, train_f1_y_false))
        # logging.debug('Epoch:[{}/{}], val f1 d:{:.2f} y:{:.2f} | d_false:{:.2f} y_false:{:.2f}'.format(e+1, args.n_epoch, val_f1_d, val_f1_y, val_f1_d_false, val_f1_y_false))
        # logging.debug('Epoch:[{}/{}], TEST f1 d:{:.2f} y:{:.2f} | d_false:{:.2f} y_false:{:.2f}'.format(e+1, args.n_epoch, test_f1_d, test_f1_y, test_f1_d_false, test_f1_y_false))
        logging.debug('Epoch:[{}/{}], train f1:{:.2f} | val f1: {:.2f} | test f1: {:.2f}'.format(e+1, args.n_epoch, train_f1, val_f1, test_f1))
        
        # save the best model
        is_best = val_f1 > best_f1
        best_f1 = max(val_f1, best_f1)
        if is_best:
            best_epoch = e
            best_test_f1 = test_f1
            checkpoint_name = 'model_best.pth.tar'
            save_checkpoint({
                'epoch': e,
                'state_dict': model.state_dict(),
                'best_f1': best_f1, 
                'optimizer': optimizer.state_dict(),
            }, is_best, filename=os.path.join(writer.log_dir, checkpoint_name), path_dir=writer.log_dir)
    
    logging.info("GILE training has finished.")
    logging.info(f"best eval f1 is {best_f1} at {best_epoch}.")
    logging.info(f"best test f1 is {best_test_f1} at {best_epoch}.")

    print('best eval f1 is {} for {}'.format(best_f1, args.name))
    print('best test f1 is {} for {}'.format(best_test_f1, args.name))


def train(model, DEVICE, optimizer, tune_loader, val_loader, test_loader, args):
    if args.now_model_name in ['GILE']:
        return train_GILE(model, DEVICE, optimizer, tune_loader, val_loader, test_loader, args)
    else:
        pass