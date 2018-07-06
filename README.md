
See updated version at http://github.com/onurgu/joint-ner-and-md-tagger

# Neural Tagger for MD and NER

This repo contains the software that was used to conduct the experiments reported
in our article titled "Improving Named Entity Recognition by Jointly Learning to 
Disambiguate Morphological Tags" [1] to be presented at [COLING 2018](http://coling2018.org).


# Training and testing

We recommend using the helper scripts for conducting experiments. The scripts named `helper-script-*`
run the experiments in the paper with given hyper parameters.


    bash ./scripts/helper-script-to-run-the-experiment-set-small-sizes.sh campaing_name | parallel -j6

For the reporting part to work, you should set up a working [`sacred`](https://github.com/IDSIA/sacred)
 environment, which is very easy if you choose a filesystem based storage. You can find an
 example of this in the helper script found in `./scripts/TRUBA` folder.

## Tag sentences

This project do not have a designated tagger script for now but you can obtain the output in `eval_dir`. 
You should provide the text in tokenized form in CoNLL format.
The script will tag both the development and testing files and produce files in `./evaluation/temp/eval_logs/`.
If you need this and want to contribute by coding and sharing it with the project,
you are welcome.

## Replication of the experiments

To reproduce the experiments reported with our model, you can use `Docker`
and build a replica of our experimentation environment.

To build:

```bash
docker build -t yourimagename:yourversion .
```

To run:
```bash
docker run -ti -v `pwd`/dataset:/opt/ner-tagger-dynet/dataset -v `pwd`/models:/opt/ner-tagger-dynet/models yourimagename:yourversion python train.py --train dataset/gungor.ner.train.small --dev dataset/gungor.ner.dev.small --test dataset/gungor.ner.test.small --word_dim 300 --word_lstm_dim 200 --word_bidirect 1 --cap_dim 100 --crf 1 --lr_method=adam --maximum-epochs 50 --char_dim 200 --char_lstm_dim 200 --char_bidirect 1 --overwrite-mappings 1 --batch-size 1
```

You should create or set permissions accordingly for ``` `pwd`/dataset ``` and ``` `pwd`/models ```.

## References

[1] Gungor, O., Uskudarli, S., Gungor, T., Improving Named Entity Recognition by Jointly Learning to 
Disambiguate Morphological Tags, 2018, COLING 2018, 19-25 August, (to appear).
