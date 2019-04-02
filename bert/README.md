# 项目结构

.
├── chinese_L-12_H-768_A-12               中文预训练模型
├── CONTRIBUTING.md
├── create_pretraining_data.py
├── evaluate-v1.1.py
├── extract_features.py
├── __init__.py
├── LICENSE
├── modeling.py
├── modeling_test.py
├── optimization.py
├── optimization_test.py
├── __pycache__
├── README.md
├── requirements.txt
├── run_classifier.py                               BERT进行分类脚本
├── run_classifier.sh
├── run_pretraining.py
├── thucnews                                         中文文本分类数据
├── sample_text.txt
├── tokenization.py
└── tokenization_test.py

# 训练数据格式

```json
4	欧典地板推出5年发展计划2010年3月27日，在欧典地板的经销商大会上，欧典地板
总裁闫培金先生宣布将全面实施“五年发展计划”。在未来的5年之内，欧典地板将全面加快
建设国内网络销售渠道，在奠定国内的主导品牌地位之后，还将开始全球化战略，开拓欧洲
、北美等国际市场。欧典地板总裁闫培金先生给这次会议定了个口号：“众望所归、再创辉
煌”。自1999年开始创业，欧典在地板行业一直扮演着一个开创者的角色。中国林产工业协
会秘书长石峰表示，欧典地板在技术开发、产品质量、售后服务等方面一直在行业里面处于
领先地位，希望欧典能够在未来的发展当中本着为消费者服务的宗旨，取得开创性的发展。
强化地板仍然是欧典目前的主销产品，是欧典技术优势比较明显、销售时间最长，系列最为
丰富的产品。在环保方面，自1999年推出第一块“船甲板”系列地板，就达到了甲醛释放量每
百克0.5mg的EO级标准；2004年欧典推出的真木纹系列也成为行业最为畅销的产品之一。本
着技术开发为先导的思路，2009年，欧典推出的新“船甲板”、“真木纹”系列再次受到市场和
消费者的欢迎。地板安装和售后服务历来是消费者最为担心和一些品牌遭到诟病的主要原因
。欧典一直以来就十分重视售后服务，把服务增值作为品牌的核心竞争力之一。欧典推出的
安装工人限量工作，提前24小时送货等举措，都保证了工人的施工质量和服务水平。“中国
未来强化地板将迎来更广阔的市场空间。”闫培金先生表示，随着全球范围的环保经济盛行
，以及大众化装修需求的持续增长，强化地板将成为最主流的产品，因此欧典将不断在产品
研发和售后服务方面持续提升，把最环保、质量最好的强化地板提供给广大消费者。同时，
还会积极与欧洲的地板生产企业积极合作，保证欧典地板的世界领先地位和品质。中国林林
产工业协会秘书长石峰先生出席本次会议并讲话。我要评论
```

左边一列是类别，右边是文本内容

# 运行脚本

```
  python run_classifier.py \            
  			--vocab_file=chinese_L-12_H-768_A-12/vocab.txt \
  			--bert_config_file=chinese_L-12_H-768_A-12/bert_config.json \
  			--init_checkpoint=chinese_L-12_H-768_A-12/bert_model.ckpt \
  			--do_train=false \
  			--train_file=./cmrc2018/data/cmrc2018_train.json \
  			--do_eval=false \
  			--do_predict=true\
  			--predict_file=./cmrc2018/data/cmrc2018_trial.json \
  			--train_batch_size=10 \
  			--learning_rate=3e-5 \
  			--num_train_epochs=3.0 \
  			--max_seq_length=300 \
  			--doc_stride=128 \
  			--output_dir=/tmp/thucnews/

```

+ `vocab_file`：词典文件，中文预训练模型里就有
+ `bert_config_file`：BERT模型的参数文件，不需要改动
+ `init_checkpoint`：初始的预训练模型里面位置
+ `train_file`：训练数据位置
+ `predict_file`：预测数据的位置
+ `train_batch_size`：训练的`batch`
+ `learning_rate`：学习率
+ `max_seq_length`：最大序列长度，*不要调得太大，可能会引起显存不够的情况*
+ `output_dir`：输出模型和输出文件的位置

# 训练

其他参数不需要改变，将运行脚本`run_squad.sh`中的`do_train`改为`true`，`do_eval`和`do_predict`改为`false`.模型训练开始,模型好的模型存于`output_dir`中

# 验证

其他参数不需要改变，将运行脚本`run_classifier.sh`中的`do_eval`改为`true`，`do_train`和`do_predict`改为`false`.

# 预测

其他参数不需要改变，将运行脚本`run_classifier.sh`中的`do_predict`改为`true`，`do_train`和`do_eval`改为`false`.根据提示输入文章，会打印类别.