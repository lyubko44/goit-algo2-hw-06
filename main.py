import requests
import re
from collections import Counter
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt


def fetch_text_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.text


def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())


def map_words(words_chunk):
    return Counter(words_chunk)


def reduce_counts(mapped_counts):
    total = Counter()
    for partial in mapped_counts:
        total.update(partial)
    return total


def map_reduce_token_frequencies(words, num_chunks=None):
    if num_chunks is None:
        num_chunks = cpu_count()
    chunk_size = len(words) // num_chunks
    chunks = [words[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]

    with Pool(num_chunks) as pool:
        mapped = pool.map(map_words, chunks)
    reduced = reduce_counts(mapped)
    return reduced


def visualize_top_words(freq_dict, top_n=10):
    most_common = freq_dict.most_common(top_n)
    words, counts = zip(*most_common)

    plt.figure(figsize=(10, 6))
    plt.barh(words[::-1], counts[::-1], color='skyblue')
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.title(f'Top {top_n} Most Frequent Words')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    url = 'https://www.gutenberg.org/files/1342/1342-0.txt'
    text = fetch_text_from_url(url)
    tokens = tokenize(text)
    word_freqs = map_reduce_token_frequencies(tokens)
    visualize_top_words(word_freqs, top_n=10)
