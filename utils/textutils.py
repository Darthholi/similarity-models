#
# COPYRIGHT Martin Holecek 2019
#

import unicodedata

import numpy as np
import re


def remove_accents(input_char):
    nfkd_form = unicodedata.normalize('NFKD', input_char)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    if len(only_ascii) <= 0:
        return nfkd_form
    return only_ascii


default_char_list = u'abcdefghijklmnopqrstuvwxyz0123456789 ,.-+:/%?$£€#()&\''
default_char_vocab = {letter: i for i, letter in enumerate(list(default_char_list))}


def text_onehot_chars(text, fixlen=None):
    text = text.lower()
    text = [remove_accents(ch) for ch in text]
    ids = [default_char_vocab[character] for character in text if character in default_char_vocab]
    ret = np.zeros((fixlen if fixlen else len(ids), len(default_char_vocab)))
    for pos, id in enumerate(ids):
        if pos >= ret.shape[0]:
            break
        ret[pos, id] = 1.0
    return ret


def base_text_features(text, features=['len', 'upper', 'lower', 'alpha', 'digit'],
                       scale=20,
                       char_vocab=default_char_vocab):
    # this is actually based on my own idea that someone else succesfully published before I got the chance :D
    def text_histogram(text, char_vocab,
                       char_vocab_make_lower=True):
        hist = [0] * len(char_vocab.keys())
        for letter in text:
            if char_vocab_make_lower:
                luse = letter.lower()
            else:
                luse = letter
            luse = remove_accents(luse)
            if luse in char_vocab:
                hist[char_vocab[luse]] += 1
        return hist
    
    def count_uppers(text):
        return sum([letter.isupper() for letter in text])
    
    def count_lowers(text):
        return sum([letter.islower() for letter in text])
    
    def count_alphas(text):
        return sum([letter.isalpha() for letter in text])
    
    def count_digits(text):
        return sum([letter.isdigit() for letter in text])
    
    use_cases = {
        'len': len,
        'upper': count_uppers,
        'lower': count_lowers,
        'alpha': count_alphas,
        'digit': count_digits,
    }
    repr = [use_cases[feature](text) for feature in features]
    if char_vocab is not None:
        repr.extend(text_histogram(text, default_char_vocab))
    
    if scale is not None:
        for i in range(len(repr)):
            repr[i] = min(repr[i] / scale, 1.0)
    
    return repr


def text_tokens(text):
    text = text.lower()
    text = remove_accents(text)
    text = re.sub("\\s", " ", text)
    text = re.sub("<br/>", " ", text)
    text = re.sub("<br />", " ", text)
    text = re.sub("<br>", " ", text)
    # text = re.sub(\"[^a-zA-Z' ]\", \"\", text)\n",
    tokens = re.split('(\W+)', text)  # text.split(' ')\n",
    tokens = [re.sub(" ", "", token) for token in tokens if token not in [' ', '']]
    return tokens


def features_from_sentence(text, values_scales, scale=20,
                           aggregate=None, pad_value=0):
    feat_list = [features_from_text(token, values_scales, scale) for token in text_tokens(text)]
    
    # So far the processing of text tokens concatenates them right away, so no batching-pading used
    # is used and so it  needs to be padded here
    def ensure_pad(array, size, beg, pad_item=[pad_value] * features_from_text_len(values_scales, scale)):
        pad_need_len = max(size - len(array), 0)
        if pad_need_len <= 0:
            return array
        if beg:
            return pad_need_len * [pad_item] + array
        else:
            return array + pad_need_len * [pad_item]
    
    if aggregate == 'sum':
        feat_list = sum(feat_list)
    if isinstance(aggregate, int):
        feat_list = ensure_pad(feat_list[:aggregate], aggregate, False)
    elif isinstance(aggregate, list):
        feat_list = ensure_pad(feat_list[:aggregate[0]], aggregate[0], False) + \
                    ensure_pad(feat_list[-aggregate[1]:], aggregate[1], True)
    return np.asarray(feat_list)


def features_from_text(text, values_scales, scale=20, char_vocab=default_char_vocab):
    try:
        xtextasval = float(text.replace(" ", "").replace("%", ""))
        xtextisval = 1.0
        assert np.isfinite(xtextasval)
    except:
        xtextasval = 0.0
        xtextisval = 0.0
    if xtextisval > 0.0:  # is actually a value
        xtextasval = [min(xtextasval / scale, 1.0) for scale in values_scales]
    else:
        xtextasval = [0.0] * len(values_scales)
    
    allfeats = base_text_features(text, scale=scale, features=['len', 'upper', 'lower', 'alpha', 'digit'],
                                  char_vocab=char_vocab)
    if len(text) <= 1:
        # just if we use the histograms for first two letters and last two letters, what to do in smaller
        text_to_handle = " " + text + " "
    else:
        text_to_handle = text
    begfeats = base_text_features(text_to_handle[0:2], scale=scale, features=['upper', 'lower', 'alpha', 'digit'],
                                  char_vocab=char_vocab)
    endfeats = base_text_features(text_to_handle[-2:0], scale=scale, features=['upper', 'lower', 'alpha', 'digit'],
                                  char_vocab=char_vocab)
    
    return allfeats + begfeats + endfeats + xtextasval + [xtextisval]


features_from_text_base_len = len(features_from_text("dummy", []))


def features_from_text_len(values_scales, scale=20):
    return features_from_text_base_len + len(values_scales)
