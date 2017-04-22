import os
import re
import codecs
from utils import create_dico, create_mapping, zero_digits
from utils import iob2, iob_iobes

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np


def load_sentences(path, lower, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """

    sentences = []
    sentence = []
    max_sentence_length = 0
    max_word_length = 0

    for line in codecs.open(path, 'r', 'utf8'):
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    # print sentence
                    # sys.exit()
                    sentences.append(sentence)
                    if len(sentence) > max_sentence_length:
                        max_sentence_length = len(sentence)
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)
            if len(word[0]) > max_word_length:
                max_word_length = len(word[0])
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
            if len(sentence) > max_sentence_length:
                max_sentence_length = len(sentence)
    return sentences, max_sentence_length, max_word_length


def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            print s_str
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')


def word_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    # words = [[(" ".join(x[0:2])).lower() if lower else " ".join(x[0:2]) for x in s] for s in sentences]
    words = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    # TODO: only roots version, but this effectively damages char embeddings.
    # words = [[x[1].split("+")[0].lower() if lower else x[1].split("+")[0] for x in s] for s in sentences]
    dico = create_dico(words)
    dico['<UNK>'] = 10000000
    word_to_id, id_to_word = create_mapping(dico)
    print "Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in words)
    )
    return dico, word_to_id, id_to_word


def char_mapping(sentences):
    """
    Create a dictionary and mapping of characters, sorted by frequency.
    """
    chars = ["".join([w[0] for w in s]) for s in sentences]
    dico = create_dico(chars)
    char_to_id, id_to_char = create_mapping(dico)
    print "Found %i unique characters" % len(dico)
    return dico, char_to_id, id_to_char


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[word[-1] for word in s] for s in sentences]
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print "Found %i unique named entity tags" % len(dico)
    return dico, tag_to_id, id_to_tag

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def cap_feature(s):
    """
    Capitalization feature:
    0 = low caps
    1 = all caps
    2 = first letter caps
    3 = one capital (not first letter)
    """

    def cap_characterization(input_s):
        if input_s.lower() == input_s:
            return 0
        elif input_s.upper() == input_s:
            return 1
        elif input_s[0].upper() == input_s[0]:
            return 2
        elif sum([x == y for (x, y) in zip(input_s.upper(), input_s)]) > 0:
            return 3

    if is_number(s):
        return 0
    elif sum([(str(digit) in s) for digit in range(0, 10)]) > 0:
        if "'" in s:
            return 1 + cap_characterization(s)
        else:
            return 1 + 4 + cap_characterization(s)
    else:
        if "'" in s:
            return 1 + 8 + cap_characterization(s)
        else:
            return 1 + 12 + cap_characterization(s)


def prepare_sentence(str_words, word_to_id, char_to_id, lower=False):
    """
    Prepare a sentence for evaluation.
    """
    def f(x): return x.lower() if lower else x
    words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
             for w in str_words]
    chars = [[char_to_id[c] for c in w if c in char_to_id]
             for w in str_words]
    caps = [cap_feature(w) for w in str_words]
    return {
        'str_words': str_words,
        'words': words,
        'chars': chars,
        'caps': caps
    }


def prepare_dataset(sentences, word_to_id, char_to_id, tag_to_id,
                    global_max_sentence_length, global_max_char_length,
                    lower=False):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """

    def f(x): return x.lower() if lower else x
    data = []

    for s in sentences:
        str_words = [w[0] for w in s]
        words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
                 for w in str_words]
        # Skip characters that are not in the training set
        chars = [[char_to_id[c] for c in w if c in char_to_id]
                 for w in str_words]
        caps = [cap_feature(w) for w in str_words]
        tags = [tag_to_id[w[-1]] for w in s]

        data.append({
            'str_words': str_words,
            'word_ids': words,
            'char_for_ids': chars,
            'char_lengths': [len(char) for char in chars],
            'cap_ids': caps,
            'tag_ids': tags,
            'sentence_lengths': len(s),
            'max_word_length_in_this_sample': max([len(x) for x in chars])
        })
    logging.info("Sorting the dataset by sentence length..")
    data_sorted_by_sentence_length = sorted(data, key=lambda x: x['sentence_lengths'])
    stats = [[x['sentence_lengths'],
              x['max_word_length_in_this_sample'],
              x['char_lengths']] for x in data]
    n_unique_words = set()
    for x in data:
        for word_id in x['word_ids']:
            n_unique_words.add(word_id)
    n_unique_words = len(n_unique_words)

    n_bins = min([9, len(sentences)])
    print "n_sentences: %d" % len(sentences)
    n_samples_to_be_binned = len(sentences)/n_bins

    print "n_samples_to_be_binned: %d" % n_samples_to_be_binned

    bins = []
    for bin_idx in range(n_bins+1):
        logging.info("Forming bin %d.." % bin_idx)
        data_to_be_binned = data_sorted_by_sentence_length[n_samples_to_be_binned*(bin_idx):n_samples_to_be_binned*(bin_idx+1)]
        if len(data_to_be_binned) == 0:
            continue
        max_sentence_length = data_to_be_binned[-1]['sentence_lengths']
        max_word_length = max([x['max_word_length_in_this_sample'] for x in data_to_be_binned])
        maxes = [max_sentence_length,
                 max_word_length]
        logging.info("%s" % maxes)
        n_samples_in_the_bin = len(data_to_be_binned)
        words_ar = np.zeros((n_samples_in_the_bin, global_max_sentence_length))
        chars_ar = np.zeros((n_samples_in_the_bin, global_max_sentence_length, global_max_char_length))
        char_lengths_ar = np.zeros((n_samples_in_the_bin, global_max_sentence_length))
        caps_ar = np.zeros((n_samples_in_the_bin, global_max_sentence_length))
        tags_ar = np.zeros((n_samples_in_the_bin, global_max_sentence_length))
        sentence_lengths_ar = np.zeros((n_samples_in_the_bin,))
        arrays_and_labels = [[words_ar, 'word_ids'],
                             [chars_ar, 'char_for_ids'],
                             [char_lengths_ar, 'char_lengths'],
                             [caps_ar, 'cap_ids'],
                             [tags_ar, 'tag_ids'],
                             [sentence_lengths_ar, 'sentence_lengths']]
        for i, d in enumerate(data_to_be_binned):
            if i % 100 == 0:
                logging.info("Sample %d is being binned" % i)
            for arr, label in arrays_and_labels:
                # logging.info("Label: %s" % label)
                # print d[label]
                if len(arr.shape) == 2:
                    arr[i,:(len(d[label]))] = d[label]
                elif len(arr.shape) == 3:
                    subarray_shape = arr[i,].shape
                    # logging.info("Subarray shape: %s" % subarray_shape)
                    arr[i,] = np.array([row + [0]*(subarray_shape[1]-len(row)) for row in d[label] + [[0]*subarray_shape[1]]*(subarray_shape[0]-len(d[label]))])
                else:
                    arr[i] = d[label]

        bin_data_dict = {label: arr for arr, label in arrays_and_labels}
        bin_data_dict['max_sentence_length'] = max_sentence_length
        bin_data_dict['max_word_length'] = max_word_length
        bins.append((bin_data_dict, maxes))

    return bins, stats, n_unique_words

def _load_and_enqueue(sess, bucket_data, n_batches, batch_size_scalar, placeholders, enqueue_op,
                      train=True):

    # TODO: shuffle the bucket_data here.

    n_sentences = len(bucket_data["sentence_lengths"])

    if train:
        new_indices = np.random.permutation(n_sentences)

        print "Reshuffling"
        for key in bucket_data.keys():
            if key in ["max_sentence_length", "max_word_length"]:
                continue
            if bucket_data[key].ndim > 1:
                bucket_data[key] = bucket_data[key][new_indices, :]
            else:
                bucket_data[key] = bucket_data[key][new_indices]

    def read_an_example(data, batch_idx):
        ret_dict = {}
        # print batch_idx
        # print batch_size_scalar
        for key in data.keys():
            if key in ["max_sentence_length", "max_word_length"]:
                continue
            # else:
            #     print "processing: %s" % key
            # try:
            # print data[key].shape
            # except AttributeError as e:
            #     print e
            lower_index = batch_idx * batch_size_scalar
            upper_index = min((batch_idx + 1) * batch_size_scalar, n_sentences)
            if data[key].ndim > 1:
                ret_dict[key] = data[key][np.arange(lower_index, upper_index), :]
            else:
                # print data[key].shape
                ret_dict[key] = data[key][np.arange(lower_index, upper_index)]
            for i in range(batch_size_scalar-(upper_index-lower_index)):
                if ret_dict[key].ndim > 1:
                    row_to_be_duplicated = ret_dict[key][0, :]
                    # print "n_sentences: %d" % n_sentences
                    # print key
                    # print ret_dict[key].shape
                    # print row_to_be_duplicated.shape
                    # print np.expand_dims(row_to_be_duplicated, axis=0).shape
                    ret_dict[key] = np.concatenate(
                        [ret_dict[key], np.expand_dims(row_to_be_duplicated, axis=0)])
                else:
                    row_to_be_duplicated = ret_dict[key][0]
                    # print "n_sentences: %d" % n_sentences
                    # print key
                    # print ret_dict[key].shape
                    # print row_to_be_duplicated.shape
                    # print np.expand_dims(row_to_be_duplicated, axis=0).shape
                    ret_dict[key] = np.concatenate([ret_dict[key], np.expand_dims(row_to_be_duplicated, axis=0)])

        # for key in ret_dict.keys():
        #     print key
        #     print ret_dict[key].shape

        return ret_dict

    for i in range(n_batches):
        given_placeholders = \
            read_an_example(bucket_data, i)

        given_placeholders['is_train'] = train

        # print given_placeholders

        # data = read_an_example()
        # print data

        feed_dict = {placeholders[key]: given_placeholders[key] for key in placeholders.keys()}

        sess.run(enqueue_op, feed_dict=feed_dict)


def augment_with_pretrained(dictionary, ext_emb_path, words):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    print 'Loading pretrained embeddings from %s...' % ext_emb_path
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = set([
        line.split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        if len(ext_emb_path) > 0
    ])

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    if words is None:
        for word in pretrained:
            if word not in dictionary:
                dictionary[word] = 0
    else:
        for word in words:
            if any(x in pretrained for x in [
                word,
                word.lower(),
                re.sub('\d', '0', word.lower())
            ]) and word not in dictionary:
                dictionary[word] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word


def calculate_global_maxes(max_sentence_lengths, max_word_lengths):
    global_max_sentence_length = 0
    global_max_char_length = 0
    for i, d in enumerate([max_sentence_lengths, max_word_lengths]):
        for label in d.keys():
            if i == 0:
                if d[label] > global_max_sentence_length:
                    global_max_sentence_length = d[label]
            elif i == 1:
                if d[label] > global_max_char_length:
                    global_max_char_length = d[label]
    return global_max_sentence_length, global_max_char_length