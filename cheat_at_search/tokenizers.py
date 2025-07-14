import Stemmer
import string

stemmer = Stemmer.Stemmer('english', maxCacheSize=0)

fold_to_ascii = dict([(ord(x), ord(y)) for x, y in zip(u"‘’´“”–-", u"'''\"\"--")])
punct_trans = str.maketrans({key: ' ' for key in string.punctuation})
all_trans = {**fold_to_ascii, **punct_trans}


def stem_word(word):
    return stemmer.stemWord(word)


def snowball_tokenizer(text):
    if type(text) == float:
        return ''
    if text is None:
        return ''
    text = text.translate(all_trans).replace("'", " ")
    split = text.lower().split()
    return [stem_word(token)
            for token in split]


def taxonomy_tokenizer(text):
    # Turn "/" into a special token that won't get stemmed
    text = text.replace('/', 'ddd')
    if type(text) == float:
        return ''
    if text is None:
        return ''
    text = text.translate(all_trans).replace("'", " ")
    split = text.lower().split()
    return [stem_word(token)
            for token in split]


def ws_tokenizer(text):
    if type(text) == float:
        return ''
    if text is None:
        return ''
    text = text.translate(all_trans)
    split = text.lower().split()
    return split


if __name__ == '__main__':
    print(snowball_tokenizer('Hello, worlds!'))
    print(ws_tokenizer('Hello, worlds!'))
