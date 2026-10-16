"""Joint SentencePiece BPE fitted exclusively to training pairs."""
from pathlib import Path
import sentencepiece as spm

PAD, UNK, BOS, EOS = 0, 1, 2, 3


def train_tokenizer(rows, prefix, vocab_size):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        sentence_iterator=(r[lang] for r in rows for lang in ('en', 'ru')),
        model_prefix=str(prefix), vocab_size=vocab_size, model_type='bpe',
        pad_id=PAD, unk_id=UNK, bos_id=BOS, eos_id=EOS,
        character_coverage=1.0, hard_vocab_limit=False, shuffle_input_sentence=False,
        num_threads=1, minloglevel=2)
    return Tokenizer(str(prefix) + '.model')


class Tokenizer:
    def __init__(self, model):
        self.sp = spm.SentencePieceProcessor(model_file=str(model))

    def __len__(self):
        return self.sp.get_piece_size()

    def encode(self, text, max_length=128):
        return [BOS] + self.sp.encode(text, out_type=int)[:max_length - 2] + [EOS]

    def decode(self, ids):
        ids = list(ids)
        if EOS in ids:
            ids = ids[:ids.index(EOS)]
        return self.sp.decode([i for i in ids if i not in (PAD, BOS)])
