
## NER Tagger in Tensorflow

This is an implementation of Lample et al. (2016) in Tensorflow which borrows several utility functions
from [their implementation](https://github.com/glample/tagger).

The model is basically a Bi-LSTM based sequence tagger which can utilize several sources of information about
each word unit like word embeddings, character based embeddings and capitalization features to obtain
the representation for that specific word unit. The model does not rely on any external data other
than having the option to initialize with pretrained word embeddings.

The following is an example command to train a model with training, development and testing datasets in
CoNLL format. Word embeddings should be in the usual text format.

You can use the '--reload' option to resume the training.

```
python train_tensorflow.py --pre_emb we-300.txt \
--train dataset/train \
--dev dataset/dev \
--test dataset/test \
--word_dim 300 --word_lstm_dim 200 --word_bidirect 1 \
--cap_dim 100 \
--crf 1 \
--lr_method=sgd-lr_0.01 \
--maximum-epochs 100 \
--char_dim 200 --char_lstm_dim 100 --char_bidirect 1 \
--overwrite-mappings 1
```

The evaluation is done separately with this command:

```
python eval_tensorflow.py --pre_emb we-300.txt \
--train dataset/train \
--dev dataset/dev \
--test dataset/test \
--word_dim 300 --word_lstm_dim 200 --word_bidirect 1 \
--cap_dim 100 \
--crf 1 \
--lr_method=sgd-lr_0.01 \
--maximum-epochs 100 \
--char_dim 200 --char_lstm_dim 100 --char_bidirect 1
```

## Tag sentences

This project do not have a designated tagger script for now but you can use eval_tensorflow.py script
to obtain the output in `eval_dir`. You should provide the text in tokenized form in CoNLL format.
The script will tag both the development and testing files and produce files in `eval_dir`.
