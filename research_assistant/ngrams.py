"""Count-based n-grams with Laplace probabilities and observed-count generation."""
from collections import Counter, defaultdict
import math
import random

class NGramLM:
    def __init__(self, n):
        if n not in (1, 2, 3):
            raise ValueError('Choose n=1, 2, or 3')
        self.n = n
        self.counts = defaultdict(Counter)

    def fit(self, documents):
        self.counts.clear()
        self.vocab = {'</s>', '<unk>'}
        for document in documents:
            words = list(document)
            self.vocab.update(words)
            padded = ['<s>'] * (self.n - 1) + words + ['</s>']
            for i in range(self.n - 1, len(padded)):
                context = tuple(padded[i-self.n+1:i]) if self.n > 1 else ()
                self.counts[context][padded[i]] += 1
        return self

    def probability(self, word, context=()):
        word = word if word in self.vocab else '<unk>'
        context = tuple(w if w in self.vocab or w == '<s>' else '<unk>' for w in context)
        context = context[-(self.n-1):] if self.n > 1 else ()
        if len(context) < self.n-1:
            context = ('<s>',) * (self.n-1-len(context)) + context
        counts = self.counts.get(context, Counter())
        return (counts[word] + 1) / (sum(counts.values()) + len(self.vocab))

    def sequence_probability(self, words):
        history = ['<s>'] * (self.n-1)
        logs = []
        for word in words:
            logs.append(math.log(self.probability(word, history)))
            history.append(word)
        return math.exp(sum(logs)), sum(logs)

    def generate(self, max_words=35, seed=42):
        rng = random.Random(seed)
        history = ['<s>'] * (self.n-1)
        output = []
        for _ in range(max_words):
            context = tuple(history[-(self.n-1):]) if self.n > 1 else ()
            counts = self.counts.get(context)
            if not counts:
                break
            word = rng.choices(list(counts), weights=list(counts.values()), k=1)[0]
            if word == '</s>':
                break
            output.append(word)
            history.append(word)
        return ' '.join(output)
