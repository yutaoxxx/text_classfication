import jieba
import torch
import pickle
class DataExample():
    '''定义数据'''
    def __init__(self, feature, label):
        self.feature = feature
        self.label = label
def cut_words(file_name, wordcount):
    features = []
    with open(file_name) as f:
        for line in f.readlines():
            line_array = line.strip('\n').split('\t')
            label = int(line_array[0])
            content = line_array[1]
            words = list(jieba.cut(content))
            for word in words:
                try:
                    wordcount[word] += 1
                except:
                    wordcount[word] = 1
    
def data_process(train_data, dev_data, threshold):
    '''数据预处理'''
    wordcount = {}
    word2ix = {}
    word2ix['__PAD__'] = 0
    word2ix['__UNK__'] = 1
    cut_words(train_data, wordcount)
    cut_words(dev_data, wordcount)
    for item in wordcount.keys():
        if wordcount[item] > threshold:
            word2ix[item] = len(word2ix)
    pickle.dump(word2ix, open('word2ix.pkl','wb'))
    return word2ix

def file2features(file_name,word2ix, max_seq_len, batch_size):
    tmp_feature = []
    tmp_label = []
    features = []
    count = 1
    with open(file_name) as f:
        for line in f.readlines():
            line_array = line.strip('\n').split('\t')
            label = int(line_array[0])
            content = line_array[1]
            words = list(jieba.cut(content))
            feature = []
            for word in words:
                try:
                    feature.append(word2ix[word])
                except:
                    feature.append(1)
            if len(feature) >= max_seq_len:
                feature = feature[:max_seq_len]
            else:
                feature = feature + [0] * (max_seq_len-len(feature))
           
            feature = torch.LongTensor(feature)
            if count % (batch_size+1) != 0:
                tmp_feature.append(feature)
                tmp_label.append(label)
            else:
                a = torch.stack(tmp_feature,0)
                label = torch.LongTensor(tmp_label)
                tmp_feature = []
                tmp_label = []
                features.append(DataExample(a, label))
            count += 1
    return features
            
