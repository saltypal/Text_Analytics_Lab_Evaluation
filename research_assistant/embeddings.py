"""Pretrained vectors streamed to a vocabulary limit; natural-text BERT pooling."""
import gzip
import hashlib
import time
from pathlib import Path
import numpy as np
import requests
from tqdm.auto import tqdm
from .preprocessing import tokens

SOURCES = {
    'Word2Vec': ('word2vec-google-news-300', 300, True),
    'GloVe': ('glove-wiki-gigaword-50', 50, False),
    'FastText': ('fasttext-wiki-news-subwords-300', 300, False),
}

def _read_binary_word(stream):
    word = bytearray()
    while True:
        char = stream.read(1)
        if not char:
            raise EOFError('Pretrained stream ended before vocabulary limit.')
        if char == b' ':
            return word.decode('utf-8', errors='replace')
        if char != b'\n':
            word.extend(char)

def load_static_vectors(name, vocabulary, cache_dir, limit=50000):
    """Keep only requested words from first `limit` pretrained entries. No training.

    FastText uses exported pretrained word vectors, so OOV subword synthesis is
    unavailable. Missing words are skipped, never replaced with random vectors.
    """
    if limit < 1:
        raise ValueError('Vocabulary limit must be positive.')
    source, dim, binary = SOURCES[name]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(('\n'.join(sorted(vocabulary))).encode()).hexdigest()[:12]
    cache = cache_dir / f'{source}-{limit}-{digest}.npz'
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            vectors = dict(zip(data['words'].tolist(), data['vectors']))
        print(f'{name}: loaded {len(vectors)} cached dataset words from {cache}')
        return vectors, dim
    url = f'https://github.com/piskvorky/gensim-data/releases/download/{source}/{source}.gz'
    start = time.perf_counter()
    print(f'{name}: streaming at most {limit:,} pretrained entries; retaining dataset words only.', flush=True)
    print(f'Source: {url}', flush=True)
    vectors = {}
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with gzip.GzipFile(fileobj=response.raw) as stream:
            header = stream.readline().decode('utf-8').strip().split()
            if len(header) != 2 or not all(x.isdigit() for x in header):
                raise ValueError(f'Unexpected embedding header: {header[:5]}')
            total, actual_dim = map(int, header)
            if actual_dim != dim:
                raise ValueError(f'Expected {dim} dimensions, received {actual_dim}')
            for _ in tqdm(range(min(limit, total)), desc=f'{name} entries'):
                if binary:
                    word = _read_binary_word(stream)
                    raw = stream.read(4*dim)
                    if len(raw) != 4*dim:
                        raise EOFError('Incomplete pretrained vector.')
                    if word in vocabulary:
                        vectors[word] = np.frombuffer(raw, dtype='<f4').copy()
                else:
                    line = stream.readline().decode('utf-8').strip().split()
                    if len(line) != dim + 1:
                        raise ValueError('Malformed pretrained text vector.')
                    if line[0] in vocabulary:
                        vectors[line[0]] = np.asarray(line[1:], dtype=np.float32)
    if not vectors:
        raise ValueError(f'{name} has no vocabulary matches; check download or increase limit.')
    np.savez_compressed(cache, words=np.array(list(vectors)), vectors=np.stack(list(vectors.values())))
    print(f'{name}: retained {len(vectors)}/{len(vocabulary)} distinct dataset words in {time.perf_counter()-start:.1f}s.', flush=True)
    return vectors, dim

def mean_embeddings(texts, vectors, dim):
    matrix = np.zeros((len(texts), dim), dtype=np.float32)
    matched, total, empty = 0, 0, 0
    for i, text in enumerate(texts):
        words = tokens(text)
        present = [vectors[w] for w in words if w in vectors]
        total += len(words)
        matched += len(present)
        if present:
            matrix[i] = np.mean(present, axis=0)
        else:
            empty += 1
    print(f'Token coverage: {matched/max(total, 1):.1%}; zero-vector documents: {empty}/{len(texts)}')
    return matrix

def bert_embeddings(texts, cache_dir, model_name='google-bert/bert-base-uncased', batch_size=16):
    import torch
    from transformers import AutoTokenizer, AutoModel
    texts = list(texts)
    digest = hashlib.sha256((model_name+'\n'+'\n'.join(texts)).encode()).hexdigest()[:16]
    cache = Path(cache_dir) / f'bert-{digest}.npy'
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        print(f'Loading cached BERT embeddings: {cache}')
        return np.load(cache)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'BERT: {model_name}, device={device}, max_length=256, masked mean pooling.', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    matrices = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(texts), batch_size), desc='BERT batches'):
            batch = tokenizer(texts[start:start+batch_size], padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
            hidden = model(**batch).last_hidden_state
            mask = batch['attention_mask'].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            matrices.append(pooled.cpu().numpy())
    result = np.concatenate(matrices)
    np.save(cache, result)
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    print(f'BERT complete: {result.shape}; cache={cache}', flush=True)
    return result
