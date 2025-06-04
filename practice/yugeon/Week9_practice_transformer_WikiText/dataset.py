# language_model/dataset.py

import torch
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import WikiText2

class LMDataset(Dataset):
    """
    WikiText-2 데이터를 불러와서, 지정된 시퀀스 길이(seq_len) 단위로
    (input_ids, target_ids) 쌍을 생성하는 Dataset 클래스입니다.
    """
    def __init__(self, split: str, vocab=None, tokenizer=None, seq_len: int = 32, min_freq: int = 5):
        """
        Args:
            split (str): 'train', 'valid', 또는 'test'
            vocab: 미리 만들어둔 vocabulary, 없으면 내부에서 생성
            tokenizer: 토크나이저 함수, 없으면 기본 'basic_english' 사용
            seq_len (int): 시퀀스 길이 (입력은 seq_len, 타겟은 seq_len개 다음 토큰)
            min_freq (int): 어휘 생성 시 최소 빈도수 임계값
        """
        super(LMDataset, self).__init__()
        # 기본 토크나이저 설정
        self.tokenizer = tokenizer if tokenizer is not None else get_tokenizer('basic_english')
        # WikiText2 데이터 로드
        data_iter = WikiText2(split=split)
        # Vocab이 주어지지 않으면 train split으로 생성
        if vocab is None:
            train_iter_for_vocab = WikiText2(split='train')
            self.vocab = build_vocab_from_iterator(self._yield_tokens(train_iter_for_vocab),
                                                   specials=['<unk>', '<pad>'],
                                                   min_freq=min_freq)
            self.vocab.set_default_index(self.vocab['<unk>'])
        else:
            self.vocab = vocab

        self.seq_len = seq_len
        # 전체 데이터를 토큰화 → 인덱스화 → 단일 리스트로 합치기
        tokens = []
        for line in data_iter:
            idxs = self.vocab(self.tokenizer(line))
            tokens.extend(idxs)
        # (input_ids, target_ids) 쌍 생성
        self.data = []
        for i in range(0, len(tokens) - seq_len):
            input_ids = tokens[i:i+seq_len]
            target_ids = tokens[i+1:i+seq_len+1]
            self.data.append((torch.tensor(input_ids, dtype=torch.long),
                              torch.tensor(target_ids, dtype=torch.long)))

    def _yield_tokens(self, data_iter):
        for line in data_iter:
            yield self.tokenizer(line)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]