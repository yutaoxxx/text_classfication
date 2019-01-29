import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F

def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    for epoch in range(1, args.epochs+1):
        if epoch == 5:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        for batch in train_iter:
            feature, target = batch.feature, batch.label
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()
            
            optimizer.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()
            steps += 1

            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == 
                                target.data).sum()
                accuracy = 100.0 * corrects / args.batch_size
                print('Batch[{}] - loss:{:.6f} acc:{:.4f}%({}/{})'.format(
                    steps, loss.data[0],accuracy, corrects, args.batch_size
                ))
            if steps % args.test_interval == 0:
                dev_acc = eval(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print(f'early stop by {args.early_stop} steps.')
            elif steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)
def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0,0
    for batch in data_iter:
        feature, target = batch.feature, batch.label
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()
        
        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)
        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    size = len(data_iter) * args.batch_size
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('Evalution-loss:{:.6f} acc:{:.4f}%({}/{})'.format(avg_loss,
                                                            accuracy,
                                                            corrects,
                                                            size))
    return accuracy

def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
