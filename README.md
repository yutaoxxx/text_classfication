# text_classification
尝试使用实现各种各样不同的文本分类方法，基于pytorch实现
数据集下载链接：https://pan.baidu.com/s/19EWZStVkwqxJANvgKk3O0g
目前实现了text-cnn模型

## 训练方法
usage: main.py    [-h] [-lr LR] [-epochs EPOCHS] [-batch-size BATCH_SIZE]
                  [-log-interval LOG_INTERVAL] [-test-interval TEST_INTERVAL]
                  [-save-interval SAVE_INTERVAL] [-early-stop EARLY_STOP]
               [-save-best SAVE_BEST] [-shuffle] [-save-dir SAVE_DIR]
               [-dropout DROPOUT] [-embed-dim EMBED_DIM]
               [-kernel-num KERNEL_NUM] [-kernel-sizes KERNEL_SIZES] [-static]
               [-eval-model]
