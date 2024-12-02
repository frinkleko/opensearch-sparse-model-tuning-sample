from typing import List, Tuple
from transformers import BertTokenizer
import collections
import warnings

warnings.filterwarnings("ignore")


class BiGramBertTokenizer(BertTokenizer):

    def __init__(self, vocab_file, bi_gram_file=None, **kwargs):
        super().__init__(vocab_file, **kwargs)
        if bi_gram_file:
            self.bi_gram_vocab = self._load_bi_gram_vocab(bi_gram_file)
        else:
            self.bi_gram_vocab = {}

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        bi_gram_file = kwargs.pop('bi_gram_file', None)
        tokenizer = super().from_pretrained(*args, **kwargs)
        if bi_gram_file:
            tokenizer.bi_gram_vocab = tokenizer._load_bi_gram_vocab(
                bi_gram_file)
        else:
            tokenizer.bi_gram_vocab = {}
        return tokenizer

    def _load_bi_gram_vocab(self, bi_gram_file):
        bi_gram_vocab = {}
        with open(bi_gram_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                tokens = line.strip().split()
                if len(tokens) != 2:
                    continue
                bi_gram = (tokens[0], tokens[1])
                bi_gram_vocab[bi_gram] = len(self.vocab) + len(bi_gram_vocab)
        print(f"Loaded {len(bi_gram_vocab)} bi-grams from {bi_gram_file}")
        self.vocab = {
            **self.vocab,
            **{
                k: v
                for k, v in bi_gram_vocab.items()
            }
        }  # Update the vocab with bi-gram tokens
        print(f"Total vocab size: {len(self.vocab)}")
        self.ids_to_tokens = collections.OrderedDict([
            (ids, tok) for tok, ids in self.vocab.items()
        ])
        return bi_gram_vocab

    def _tokenize(self,
                  text,
                  split_special_tokens=False) -> List[Tuple[str, ...]]:
        split_tokens = []
        if self.do_basic_tokenize:
            tokens = self.basic_tokenizer.tokenize(
                text,
                never_split=self.all_special_tokens
                if not split_special_tokens else None)
            for i, token in enumerate(tokens):
                if i < len(tokens) - 1:
                    bi_gram = (token, tokens[i + 1])
                    if bi_gram in self.bi_gram_vocab:
                        split_tokens.append(bi_gram)
                        continue
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append((token, ))
                else:
                    split_tokens += [
                        (t, ) for t in self.wordpiece_tokenizer.tokenize(token)
                    ]
        else:
            split_tokens = [(t, )
                            for t in self.wordpiece_tokenizer.tokenize(text)]
        return split_tokens

    def _convert_token_to_id(self, token):
        if token in self.bi_gram_vocab:
            return self.bi_gram_vocab[token]
        else:
            token = token[0]
            return self.vocab[token]

    def _convert_id_to_token(self, index):
        if index in self.bi_gram_vocab.values():
            return self.ids_to_tokens[index]
        else:
            return self.ids_to_tokens[index]


if __name__ == '__main__':
    text = "This is a sample text 7 km."

    print('Bi-Gram Bert Tokenizer')
    tokenizer = BiGramBertTokenizer.from_pretrained(
        'bert-base-uncased', bi_gram_file='bigrams_vocab.txt')
    
    tokens = tokenizer.tokenize(text)
    print(tokens)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(ids)
    new_tokens = tokenizer.convert_ids_to_tokens(ids)
    print(new_tokens)

    print('Original Bert Tokenizer')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens = tokenizer.tokenize(text)
    print(tokens)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(ids)
    new_tokens = tokenizer.convert_ids_to_tokens(ids)
    print(new_tokens)
