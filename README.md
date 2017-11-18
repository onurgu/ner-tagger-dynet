
# Neural NER Tagger in Dynet

This is a re-implementation of Lample et al. (2016) (see [1]) in Dynet which borrows several utility functions
from [their implementation](https://github.com/glample/tagger).

The model is basically a Bi-LSTM based sequence tagger which can utilize several sources of information about
each word unit like word embeddings, character based embeddings and capitalization features to obtain
the representation for that specific word unit. The model does not rely on any external data other
than having the option to initialize with pretrained word embeddings.

The following is an example command to train a model with training, development and testing datasets in
CoNLL format. Word embeddings should be in the usual text format.

You can use the `--reload` and `--model_path` options to resume the training.

```
python train.py --pre_emb we-300.txt 
--train dataset/gungor.ner.train.small 
--dev dataset/gungor.ner.dev.small 
--test dataset/gungor.ner.test.small 
--word_dim 300 --word_lstm_dim 200 --word_bidirect 1 
--cap_dim 100 
--crf 1 
--lr_method=adam 
--maximum-epochs 50 
--char_dim 200 --char_lstm_dim 200 --char_bidirect 1 
--overwrite-mappings 1 
--batch-size 1 
```

# Neural NER Tagger For Morphologically Rich Languages

If you add `--morpho_tag_dim` option, 

```
--morpho_tag_dim 100
```

this becomes the reference implementation for Gungor et al. (2017) [2] which describes
a method for incorporating morphological tags which turn out to be important for
morphologicall rich languages like Turkish, Czech, Finnish, etc.

## Tag sentences

This project do not have a designated tagger script for now but you can obtain the output in `eval_dir`. 
You should provide the text in tokenized form in CoNLL format.
The script will tag both the development and testing files and produce files in `eval_dir`.
If you need this and want to contribute by coding and sharing it with the project,
you are welcome.

## References

[1] Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K., & Dyer, C. (2016). Neural Architectures for Named Entity Recognition. In Proceedings of NAACL-HLT (pp. 260-270).

[2] Morphological Embeddings for Named Entity Recognition in Morphologically Rich Languages
O Gungor, E Yildiz, S Uskudarli, T Gungor - arXiv preprint arXiv:1706.00506, 2017
