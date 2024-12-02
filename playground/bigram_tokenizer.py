from typing import List
from transformers import BertTokenizer

class BiGramBertTokenizer(BertTokenizer):
    def __init__(self, vocab_file, bi_gram_file=None, **kwargs):
        super().__init__(vocab_file, **kwargs)
        if bi_gram_file:
            self.bi_gram_vocab = self._load_bi_gram_vocab(bi_gram_file)
            self.bi_gram_ids = {v: k for k, v in self.vocab.items()}
        else:
            self.bi_gram_vocab = {}
            self.bi_gram_ids = {}

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        bi_gram_file = kwargs.pop('bi_gram_file', None)
        tokenizer = super().from_pretrained(*args, **kwargs)
        if bi_gram_file:
            tokenizer.bi_gram_vocab = tokenizer._load_bi_gram_vocab(bi_gram_file)
            tokenizer.bi_gram_ids = {v: k for k, v in tokenizer.vocab.items()}
        else:
            tokenizer.bi_gram_vocab = {}
            tokenizer.bi_gram_ids = {}
        return tokenizer

    def _load_bi_gram_vocab(self, bi_gram_file):
        bi_gram_vocab = {}
        with open(bi_gram_file, 'r', encoding='utf-8') as f:
            for line in f:
                # each line is 'token1 token2' bi_gram
                tokens = line.strip().split()
                if len(tokens) != 2:
                    continue
                bi_gram_vocab[(tokens[0], tokens[1])] = tokens[0] + '_' + tokens[1]
        return bi_gram_vocab

    def _tokenize(self, text, split_special_tokens=False):
        split_tokens = []
        if self.do_basic_tokenize:
            tokens = self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens if not split_special_tokens else None)
            for i, token in enumerate(tokens):
                if i < len(tokens) - 1:
                    bi_gram = (token, tokens[i + 1])
                    if bi_gram in self.bi_gram_vocab:
                        split_tokens.append(self.bi_gram_vocab[bi_gram])
                        continue
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens


    ############# convert to id is still not working properly ################

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        if token in self.bi_gram_ids:
            return self.vocab[token]
        else:
            return super()._convert_token_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.bi_gram_ids.values():
            token = [t for t, i in self.bi_gram_ids.items() if i == index][0]
            return token
        else:
            return super()._convert_id_to_token(index)



if __name__ == '__main__':
    tokenizer = BiGramBertTokenizer.from_pretrained('bert-base-uncased', bi_gram_file='bigrams_vocab.txt')
    text = "This is a sample text 7 km."
    tokens = tokenizer.tokenize(text)
    print(tokens)  # ['this_is', 'is_a', 'a_sample', 'sample', 'text', '7_km', 'km', '.']

    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(ids) 

    new_tokens = tokenizer.convert_ids_to_tokens(ids)
    print(new_tokens) 

