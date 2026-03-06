import os
import sys
import time
import pickle
import argparse
import collections
import re
import urllib.request

import numpy as np

EMBEDDING_DIM   = 100
WINDOW_SIZE     = 5
NEG_SAMPLES     = 5
MIN_COUNT       = 2
SUBSAMPLE_T     = 1e-4
LEARNING_RATE   = 0.025
EPOCHS          = 10
MAX_TOKENS      = 17_000_000
NOISE_DIST_EXP  = 0.75
NEG_TABLE_SIZE  = 100_000_000


def download_gutenberg(book_id, output_path=None, max_tokens=None):
    output_path = output_path or f"gutenberg_{book_id}.txt"

    if not os.path.exists(output_path):
        text = None
        for url in [
    f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
    f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
    f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
]:
            try:
                with urllib.request.urlopen(url, timeout=15) as r:
                    text = r.read().decode("utf-8", errors="replace")
                break  
            except Exception:
                continue

        if text is None:
            raise RuntimeError(f"Could not download book {book_id}.")

        start = re.search(r"\*\*\* ?START OF.*?\*\*\*", text, re.IGNORECASE)
        if start:
            text = text[start.end():]

        end = re.search(r"\*\*\* ?END OF.*?\*\*\*", text, re.IGNORECASE)
        if end:
            text = text[:end.start()]

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text.strip())
      
    with open(output_path, encoding="utf-8") as f:
        tokens = re.findall(r"[a-z]+", f.read().lower())

    if max_tokens:
        tokens = tokens[:max_tokens]

    return tokens


def build_vocab(tokens, min_count=MIN_COUNT):
    counter = collections.Counter(tokens)
    vocab = [(w, c) for w, c in counter.items() if c >= min_count]
    vocab.sort(key=lambda x: -x[1])

    idx2word = [w for w, _ in vocab]
    word2idx = {w: i for i, w in enumerate(idx2word)}
    freqs    = np.array([c for _, c in vocab], dtype=np.float64)
    return word2idx, idx2word, freqs


def subsample(tokens, word2idx, freqs, t=SUBSAMPLE_T):

    total = freqs.sum()
    keep_prob = np.minimum(np.sqrt(t / (freqs / total)), 1.0)

    corpus = []
    for w in tokens:
        idx = word2idx.get(w)
        if idx is None:
            continue
        if np.random.random() < keep_prob[idx]:
            corpus.append(idx)
    return corpus



def build_neg_table(freqs, table_size=NEG_TABLE_SIZE, exp=NOISE_DIST_EXP):

    powered = freqs ** exp
    powered /= powered.sum()
    counts  = np.round(powered * table_size).astype(np.int64)

    table = np.repeat(np.arange(len(freqs)), counts)
    np.random.shuffle(table)
    return table


class Word2VecModel:

    def __init__(self, vocab_size, dim):
        self.V   = vocab_size
        self.d   = dim
        bound    = 0.5 / dim
        self.W_in  = np.random.uniform(-bound, bound, (vocab_size, dim))
        self.W_out = np.zeros((vocab_size, dim), dtype=np.float64)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"W_in": self.W_in, "W_out": self.W_out,
                         "V": self.V, "d": self.d}, f)
        print(f"[model] saved at {path}")

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        m = cls.__new__(cls)
        m.V, m.d = data["V"], data["d"]
        m.W_in   = data["W_in"]
        m.W_out  = data["W_out"]
        return m



def sigmoid(x):

    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def sgns_step(center_idx, context_idx, neg_indices, model, lr):

    h = model.W_in[center_idx]
    score_p  = np.dot(h, model.W_out[context_idx])
    scores_n = model.W_out[neg_indices] @ h

    sig_p = sigmoid(score_p)
    sig_n = sigmoid(scores_n)

    loss = -np.log(sig_p + 1e-10) - np.sum(np.log(1.0 - sig_n + 1e-10))


    err_p = sig_p - 1.0
    err_n = sig_n

    grad_out_pos = err_p * h
    grad_out_neg = err_n[:, None] * h[None, :]

    grad_h = (err_p * model.W_out[context_idx]
              + (err_n[:, None] * model.W_out[neg_indices]).sum(axis=0))


    model.W_out[context_idx] -= lr * grad_out_pos
    model.W_out[neg_indices] -= lr * grad_out_neg
    model.W_in[center_idx]   -= lr * grad_h

    return loss




def train(model, corpus, neg_table,
          epochs=EPOCHS, window=WINDOW_SIZE, K=NEG_SAMPLES, lr0=LEARNING_RATE):

    n_tokens    = len(corpus)
    total_steps = epochs * n_tokens
    step        = 0
    epoch_losses = []

    for epoch in range(epochs):
        t0         = time.time()
        epoch_loss = 0.0
        n_pairs    = 0

        for t in range(n_tokens):
            lr    = max(lr0 * (1.0 - step / total_steps), lr0 * 1e-4)
            step += 1

            center_idx = corpus[t]
            c = np.random.randint(1, window + 1)

            for j in range(max(0, t - c), min(n_tokens, t + c + 1)):
                if j == t:
                    continue
                context_idx = corpus[j]

                negs = []
                while len(negs) < K:
                    cand = neg_table[np.random.randint(len(neg_table))]
                    if cand != center_idx and cand != context_idx:
                        negs.append(cand)
                neg_indices = np.array(negs, dtype=np.int64)

                loss = sgns_step(center_idx, context_idx, neg_indices, model, lr)
                epoch_loss += loss
                n_pairs    += 1

            if t > 0 and t % 100_000 == 0:
                elapsed = time.time() - t0
                print(f"  epoch {epoch+1}/{epochs}  "
                      f"{100.0*t/n_tokens:5.1f}%  "
                      f"avg_loss={epoch_loss/n_pairs:.4f}  "
                      f"lr={lr:.6f}  elapsed={elapsed:.0f}s", flush=True)

        avg = epoch_loss / max(n_pairs, 1)
        epoch_losses.append(avg)
        print(f"[epoch {epoch+1}] avg_loss={avg:.4f}  time={time.time()-t0:.0f}s")

    return epoch_losses


def most_similar(word, word2idx, idx2word, W, topn=10):

    if word not in word2idx:
        print(f"'{word}' not in vocabulary")
        return []
    idx = word2idx[word]
    vec = W[idx]
    norms  = np.linalg.norm(W, axis=1, keepdims=True) + 1e-10
    W_norm = W / norms
    sims   = W_norm @ (vec / (np.linalg.norm(vec) + 1e-10))
    results = []
    for i in np.argsort(-sims):
        if i == idx:
            continue
        results.append((idx2word[i], float(sims[i])))
        if len(results) >= topn:
            break
    return results


def analogy(pos1, neg1, pos2, word2idx, idx2word, W, topn=5):

    for w in (pos1, neg1, pos2):
        if w not in word2idx:
            print(f"'{w}' not in vocabulary"); return []

    norms  = np.linalg.norm(W, axis=1, keepdims=True) + 1e-10
    W_norm = W / norms
    query  = (W_norm[word2idx[pos1]]
              - W_norm[word2idx[neg1]]
              + W_norm[word2idx[pos2]])
    query /= np.linalg.norm(query) + 1e-10

    sims    = W_norm @ query
    exclude = {word2idx[w] for w in (pos1, neg1, pos2)}
    top     = []
    for i in np.argsort(-sims):
        if i not in exclude:
            top.append((idx2word[i], float(sims[i])))
        if len(top) >= topn:
            break
    return top



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="text8")
    parser.add_argument("--epochs", type=int,        default=EPOCHS)
    parser.add_argument("--dim",    type=int,        default=EMBEDDING_DIM)
    parser.add_argument("--eval",   action="store_true")
    parser.add_argument("--model",  default="w2v_model.pkl")
    parser.add_argument("--vocab",  default="w2v_vocab.pkl")

    args, _ = parser.parse_known_args()

    np.random.seed(42)

    all_tokens = []
    for book_id in [1342, 2701, 84, 1661, 11, 1400, 98, 345, 74, 76, 120, 1184, 2600, 1260, 35, 36]:
        all_tokens += download_gutenberg(book_id)
    tokens = all_tokens[:MAX_TOKENS]
    print(f"[dataset] total tokens: {len(tokens):,}")

    word2idx, idx2word, freqs = build_vocab(tokens, MIN_COUNT)
    corpus = subsample(tokens, word2idx, freqs, SUBSAMPLE_T)
    neg_table = build_neg_table(freqs, NEG_TABLE_SIZE, NOISE_DIST_EXP)
    
    model = Word2VecModel(vocab_size=len(idx2word), dim=args.dim)


    train(model, corpus, neg_table,
          epochs=args.epochs, window=WINDOW_SIZE, K=NEG_SAMPLES, lr0=LEARNING_RATE)

    model.save(args.model)
    with open(args.vocab, "wb") as f:
        pickle.dump({"word2idx": word2idx, "idx2word": idx2word, "freqs": freqs}, f)
    print(f"[vocab] saved -> {args.vocab}")

    for probe in ["girl", "town", "school", "river"]:
        nbrs  = most_similar(probe, word2idx, idx2word, model.W_in, topn=5)
        print(f"  {probe:10s} -> {[w for w, _ in nbrs]}")


if __name__ == "__main__":
    main()
