import os
import argparse
import model
import train
import dataset
import datetime
import pickle
import torch
parser = argparse.ArgumentParser(description='CNN text classifier')

parser.add_argument('-lr', type=float, default=0.001)
parser.add_argument('-epochs', type=int, default=256)
parser.add_argument('-batch-size', type=int, default=64)
parser.add_argument('-log-interval', type=int, default=1)
parser.add_argument('-test-interval', type=int, default=100)
parser.add_argument('-save-interval', type=int, default=500)
parser.add_argument('-early-stop', type=int, default=1000)
parser.add_argument('-save-best', type=bool, default=True)
parser.add_argument('-shuffle', action='store_true', default=False)
parser.add_argument('-save-dir', type=str, default='snapshot')


parser.add_argument('-dropout', type=float, default=0.5)
parser.add_argument('-embed-dim', type=int, default=128)
parser.add_argument('-kernel-num', type=int, default=100)
parser.add_argument('-kernel-sizes', type=str, default='3,4,5')
parser.add_argument('-static', action='store_true', default=False)
parser.add_argument('-eval-model', action='store_true', default=False)

args = parser.parse_args()


#word2ix = dataset.data_process('./data/train.tsv', './data/dev.tsv', 50)
word2ix = pickle.load(open('word2ix.pkl','rb'))
train_iter = dataset.file2features('./data/train.tsv', word2ix, 500, args.batch_size)
dev_iter = dataset.file2features('./data/dev.tsv', word2ix, 500, args.batch_size)


args.embed_num = len(word2ix)
args.class_num = 4
args.cuda = True
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

cnn = model.CNN_Text(args)
if args.eval_model:
    count = 0
    bz = 0
    cnn.load_state_dict(torch.load('./snapshot/2019-01-29_11-24-17/best_steps_1200.pt'))
    for i in dev_iter:
        logits = cnn(i.feature)
        predit = torch.max(logits,1)[1]
        grant_truth = i.label
        for j in range(args.batch_size):
            if predit[j].item() == grant_truth[j].item():
                count += 1
        bz += 1
    print('acc = {}'.format(100.0 * count/(64 * bz)))
else:
    train.train(train_iter, dev_iter, cnn, args)
