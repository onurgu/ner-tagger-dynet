import optparse
import sys
from collections import OrderedDict

import os
import re
import codecs
import numpy as np
#import theano


models_path = "./models"
eval_path = "./evaluation"
eval_temp = os.path.join(eval_path, "temp")
eval_script = os.path.join(eval_path, "conlleval")


def get_name(parameters):
    """
    Generate a model name from its parameters.
    """
    l = []
    for k, v in parameters.items():
        if (type(v) is str or type(v) is unicode) and "/" in v:
            l.append((k, v[::-1][:v[::-1].index('/')][::-1]))
        else:
            l.append((k, v))
    name = ",".join(["%s=%s" % (k, str(v).replace(',', '')) for k, v in l])
    return "".join(i for i in name if i not in "\/:*?<>|")

def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)


def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True


def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags


def insert_singletons(words, singletons, p=0.5):
    """
    Replace singletons by the unknown word with a probability p.
    """
    new_words = []
    for word in words:
        if word in singletons and np.random.uniform() < p:
            new_words.append(0)
        else:
            new_words.append(word)
    return new_words


def pad_word_chars(words):
    """
    Pad the characters of the words in a sentence.
    Input:
        - list of lists of ints (list of words, a word being a list of char indexes)
    Output:
        - padded list of lists of ints
        - padded list of lists of ints (where chars are reversed)
        - list of ints corresponding to the index of the last character of each word
    """
    max_length = max([len(word) for word in words])
    char_for = []
    char_rev = []
    char_pos = []
    for word in words:
        padding = [0] * (max_length - len(word))
        char_for.append(word + padding)
        char_rev.append(word[::-1] + padding)
        char_pos.append(len(word) - 1)
    return char_for, char_rev, char_pos

def create_input(data, parameters, add_label, singletons=None):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    words = data['words']
    chars = data['chars']
    if singletons is not None:
        words = insert_singletons(words, singletons)
    if parameters['cap_dim']:
        caps = data['caps']
    char_for, char_rev, char_pos = pad_word_chars(chars)
    input = []

    if parameters['word_dim']:
        input.append(words)

    if parameters['char_dim']:
        input.append(char_for)
        if parameters['ch_b']:
            input.append(char_rev)
        input.append(char_pos)

    if parameters['cap_dim']:
        input.append(caps)
    # print input
    if add_label:
        input.append(data['tags'])
    # print input
    return input


def evaluate(parameters, f_eval, raw_sentences, parsed_sentences,
             id_to_tag, dictionary_tags, predict_and_exit_filename=""):
    """
    Evaluate current model using CoNLL script.
    """
    n_tags = len(id_to_tag)
    predictions = []
    count = np.zeros((n_tags, n_tags), dtype=np.int32)

    activation_values_dump_files = dict()
    if predict_and_exit_filename:
        activation_values_dump_files['char_lstm_for'] =\
            open(os.path.join("./models/", get_name(parameters), "activation_values_char_lstm_for.dat"), "w")

    for raw_sentence, parsed_sentence in zip(raw_sentences, parsed_sentences):
        input = create_input(parsed_sentence, parameters, False)
        eval_result_for_input = f_eval(*input)
        if parameters['crf']:
            # y_preds = np.array(eval_result_for_input[0])[1:-1]
            y_preds = np.array(eval_result_for_input)[1:-1]
        else:
            y_preds = eval_result_for_input.argmax(axis=1)
        y_reals = np.array(parsed_sentence['tags']).astype(np.int32)
        assert len(y_preds) == len(y_reals)
        with open("erratic-predictions.txt", "a") as f:
            if np.any((y_preds >= n_tags) * (y_preds < 0)):
                f.write(str(y_preds) + "\n")
                y_preds[(y_preds >= n_tags) * (y_preds < 0)] = 0
        p_tags = [id_to_tag[y_pred] for y_pred in y_preds]
        r_tags = [id_to_tag[y_real] for y_real in y_reals]
        if parameters['t_s'] == 'iobes':
            p_tags = iobes_iob(p_tags)
            r_tags = iobes_iob(r_tags)
        for i, (y_pred, y_real) in enumerate(zip(y_preds, y_reals)):
            new_line = " ".join(raw_sentence[i][:-1] + [r_tags[i], p_tags[i]])
            predictions.append(new_line)
            count[y_real, y_pred] += 1
        predictions.append("")

    # Write predictions to disk and run CoNLL script externally
    eval_id = np.random.randint(1000000, 2000000)
    output_path = os.path.join(eval_temp, "eval.%i.output" % eval_id)
    scores_path = os.path.join(eval_temp, "eval.%i.scores" % eval_id)
    with codecs.open(output_path, 'w', 'utf8') as f:
        f.write("\n".join(predictions))
    if predict_and_exit_filename:
        with codecs.open(predict_and_exit_filename, 'w', 'utf8') as f:
            f.write("\n".join(predictions))
        os.system("%s < %s > %s" % (eval_script, output_path, predict_and_exit_filename+".scores"))
        # CoNLL evaluation results
        eval_lines = [l.rstrip() for l in codecs.open(predict_and_exit_filename+".scores", 'r', 'utf8')]
        for line in eval_lines:
            print line
    else:
        os.system("%s < %s > %s" % (eval_script, output_path, scores_path))

        # CoNLL evaluation results
        eval_lines = [l.rstrip() for l in codecs.open(scores_path, 'r', 'utf8')]
        for line in eval_lines:
            print line

    # Remove temp files
    # os.remove(output_path)
    # os.remove(scores_path)

    # Confusion matrix with accuracy for each tag
    print ("{: >2}{: >15}{: >7}%s{: >9}" % ("{: >15}" * n_tags)).format(
        "ID", "NE", "Total",
        *([id_to_tag[i] for i in xrange(n_tags)] + ["Percent"])
    )
    for i in xrange(n_tags):
        print ("{: >2}{: >15}{: >7}%s{: >9}" % ("{: >15}" * n_tags)).format(
            str(i), id_to_tag[i], str(count[i].sum()),
            *([count[i][j] for j in xrange(n_tags)] +
              ["%.3f" % (count[i][i] * 100. / max(1, count[i].sum()))])
        )

    # Global accuracy
    print "%i/%i (%.5f%%)" % (
        count.trace(), count.sum(), 100. * count.trace() / max(1, count.sum())
    )

    # F1 on all entities
    return float(eval_lines[1].strip().split()[-1])


def read_args(evaluation=False, args_as_a_list=sys.argv[1:]):
    optparser = optparse.OptionParser()
    optparser.add_option(
        "-T", "--train", default="",
        help="Train set location"
    )
    optparser.add_option(
        "-d", "--dev", default="",
        help="Dev set location"
    )
    optparser.add_option(
        "-t", "--test", default="",
        help="Test set location"
    )
    optparser.add_option(
        "-s", "--tag_scheme", default="iobes",
        help="Tagging scheme (IOB or IOBES)"
    )
    optparser.add_option(
        "-l", "--lower", default="0",
        type='int', help="Lowercase words (this will not affect character inputs)"
    )
    optparser.add_option(
        "-z", "--zeros", default="0",
        type='int', help="Replace digits with 0"
    )
    optparser.add_option(
        "-c", "--char_dim", default="25",
        type='int', help="Char embedding dimension"
    )
    optparser.add_option(
        "-C", "--char_lstm_dim", default="25",
        type='int', help="Char LSTM hidden layer size"
    )
    optparser.add_option(
        "-b", "--char_bidirect", default="1",
        type='int', help="Use a bidirectional LSTM for chars"
    )
    optparser.add_option(
        "-w", "--word_dim", default="100",
        type='int', help="Token embedding dimension"
    )
    optparser.add_option(
        "-W", "--word_lstm_dim", default="100",
        type='int', help="Token LSTM hidden layer size"
    )
    optparser.add_option(
        "-B", "--word_bidirect", default="1",
        type='int', help="Use a bidirectional LSTM for words"
    )
    optparser.add_option(
        "-p", "--pre_emb", default="",
        help="Location of pretrained embeddings"
    )
    optparser.add_option(
        "-A", "--all_emb", default="0",
        type='int', help="Load all embeddings"
    )
    optparser.add_option(
        "-a", "--cap_dim", default="0",
        type='int', help="Capitalization feature dimension (0 to disable)"
    )
    optparser.add_option(
        "-f", "--crf", default="1",
        type='int', help="Use CRF (0 to disable)"
    )
    optparser.add_option(
        "-D", "--dropout", default="0.5",
        type='float', help="Droupout on the input (0 = no dropout)"
    )
    optparser.add_option(
        "-L", "--lr_method", default="sgd-lr_.005",
        help="Learning method (SGD, Adadelta, Adam..)"
    )
    optparser.add_option(
        "-r", "--reload", default="0",
        type='int', help="Reload the last saved model"
    )
    optparser.add_option(
        "--skip-testing", default="0",
        type='int',
        help="Skip the evaluation on test set (because dev and test sets are the same and thus testing is irrelevant)"
    )
    optparser.add_option(
        "--predict-and-exit-filename", default="",
        help="Used with '--reload 1', the loaded model is used for predicting on the test set and the results are written to the filename"
    )
    optparser.add_option(
        "--overwrite-mappings", default="0",
        type='int', help="Explicitly state to overwrite mappings"
    )
    optparser.add_option(
        "--maximum-epochs", default="1000",
        type='int', help="Maximum number of epochs"
    )
    optparser.add_option(
        "--batch-size", default="5",
        type='int', help="Number of samples in one epoch"
    )
    if evaluation:
        optparser.add_option(
            "--run-for-all-checkpoints", default="0",
            type='int', help="run evaluation for all checkpoints"
        )
    opts = optparser.parse_args(args_as_a_list)[0]
    return opts


def form_parameters_dict(opts):
    parameters = OrderedDict()
    parameters['t_s'] = opts.tag_scheme
    parameters['lower'] = opts.lower == 1
    parameters['zeros'] = opts.zeros == 1
    parameters['char_dim'] = opts.char_dim
    parameters['char_lstm_dim'] = opts.char_lstm_dim
    parameters['ch_b'] = opts.char_bidirect == 1

    parameters['word_dim'] = opts.word_dim
    parameters['word_lstm_dim'] = opts.word_lstm_dim
    parameters['w_b'] = opts.word_bidirect == 1
    parameters['pre_emb'] = opts.pre_emb
    parameters['all_emb'] = opts.all_emb == 1
    parameters['cap_dim'] = opts.cap_dim
    parameters['crf'] = opts.crf == 1
    parameters['dropout'] = opts.dropout
    parameters['lr_method'] = opts.lr_method

    return parameters