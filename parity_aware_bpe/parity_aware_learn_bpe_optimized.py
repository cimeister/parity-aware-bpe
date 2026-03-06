#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Rico Sennrich, Negar Foroutan

""" Parity-aware BPE learns a tokenization that ensures parity in token lengths across languages on a multi-parallel development set.
Unlike standard BPE, which optimizes merges based on a single corpus, this approach explicitly considers cross-lingual fairness during the tokenization process.
"""

from __future__ import unicode_literals

import os
import sys
import inspect
import codecs
import re
import copy
import argparse
import warnings
import tempfile
import functools
import operator
import numpy
import logging
import numpy as np
import json
import io
import gc

from multiprocessing import Pool, cpu_count
from collections import defaultdict, Counter, deque
from contextlib import contextmanager

from tokenizers.pre_tokenizers import Whitespace, ByteLevel, Split
from tokenizers import pre_tokenizers, Regex
from tokenizers import Tokenizer, models, decoders, normalizers
from tokenizers.processors import TemplateProcessing
SPECIAL_TOKENS = ["<s>", "<pad>", "</s>", "<unk>"]

# GPT-4 style split pattern used by --apertus-formatting
APERTUS_SPLIT_PATTERN = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"

APERTUS_V2_SPLIT_PATTERN = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+|\s*[\r\n]+|\s+(?!\S)|\s+"

# Create a logger
logger = logging.getLogger(__name__)

# Default pre-tokenizer (used when no explicit pre_tokenizer is passed)
DEFAULT_PRE_TOKENIZER = pre_tokenizers.Sequence([Whitespace(), ByteLevel(use_regex=False, add_prefix_space=True)])
decoder = decoders.ByteLevel()

# Worker-level state for multiprocessing (set via pool initializer)
_worker_pre_tokenizer = None

def _init_worker(pre_tok):
    """Pool initializer: sets worker-level pre_tokenizer so spawn-based workers use the correct one."""
    global _worker_pre_tokenizer
    _worker_pre_tokenizer = pre_tok


try:
    from tqdm import tqdm
    tqdm.monitor_interval = 0
except ImportError:
    def tqdm(iterator, *args, **kwargs):
        return iterator


def set_logger(verbose=True):
    if verbose:
        level = logging.INFO
    else: 
        level = logging.WARN
    logger.setLevel(level)

    # Create console handler and set level to info
    ch = logging.StreamHandler()
    ch.setLevel(level)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(ch)


def _log_memory(label=""):
    """Log current RSS memory usage (Linux only)."""
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    rss_kb = int(line.split()[1])
                    logger.info(f"[MEM {label}] RSS = {rss_kb / 1024 / 1024:.1f} GB")
                    return
    except Exception:
        pass


def create_parser(subparsers=None):

    if subparsers:
        parser = subparsers.add_parser('parity-aware-learn-bpe',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="learn Parity-aware BPE-based word segmentation")
    else:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="learn Parity-aware BPE-based word segmentation")

    parser.add_argument(
        '--variant', type=str, default='base',
        help="Partiy-aware BPE variant, either 'base' or 'window'. Default: %(default)s")
    parser.add_argument(
        '--input', '-i', type=str, default=None, nargs='*',
        metavar='PATHS',
        help="Input texts or parquet files (default: standard input).")
    parser.add_argument(
        '--dev', '-d', type=argparse.FileType('r'), nargs='*',
        metavar='PATHS',
        help="Development texts (are used for parity computation).")
    parser.add_argument(
        '--ratio', '-r', type=float, nargs='*',
        help="Desired ratio of compression (comparing to pre-tokenized length) per input language. Can be used for parity computation in lieu of development set.")
    parser.add_argument(
        '--output', '-o', type=str, default=None,
        metavar='PATH',
        help="Output file for BPE codes (default: standard output)")
    parser.add_argument(
        '--json-output', '-j', type=str, default=None,
        metavar='PATH',
        help="Output file for Hugging Face tokenizer.json (default: None)")
    parser.add_argument(
        '--symbols', '-s', type=int, default=10000,
        help="Create this many new symbols (each representing a character n-gram) (default: %(default)s)")
    parser.add_argument(
        '--global-merges', '-g', type=int, default=0,
        help="For first INT merge operations, do merge based on global statistics instead of parity-driven language-specifc ones (default: %(default)s)")
    parser.add_argument(
    '--config', type=str, default=None,
    metavar='PATH',
    help="JSON config file specifying inputs and ratios. Expected format: {\"languages\": [{\"name\": \"en\", \"input\": \"path/to/en.txt\", \"dev\": \"path/to/en.dev.txt\", \"ratio\": 1.0}, ...]}")
    parser.add_argument(
        '--min-frequency', type=int, default=2, metavar='FREQ',
        help='Stop if no symbol pair has frequency >= FREQ (default: %(default)s)')
    parser.add_argument(
        '--preload', type=argparse.FileType('r'), default=None,
        metavar='PATH',
        help="Preload merges from BPE file (default: None). Can be used to continue learning with different settings (e.g. without whitespace pre-tokenization for SuperBPE).")
    parser.add_argument(
        '--pretokenize', type=str, default=['whitespace', 'bytelevel'], nargs='*',
        choices=['whitespace', 'bytelevel'],
        help="Pre-tokenizer(s) to apply. Ignored with --apertus-formatting. (default: %(default)s)")
    parser.add_argument('--dict-input', action="store_true",
        help="If set, input file is interpreted as a dictionary where each line contains a word-count pair")
    parser.add_argument(
        '--text-column', type=str, default='text',
        metavar='COL',
        help="Column name containing text data in parquet input files (default: %(default)s)")
    parser.add_argument(
        '--total-symbols', '-t', action="store_true",
        help="subtract number of characters from the symbols to be generated (so that '--symbols' becomes an estimate for the total number of symbols needed to encode text).")
    parser.add_argument(
        '--window-size', '-w', type=int, default=100,
        help="Size of the context window for the moving-window balancing variant of parity-aware BPE (default: %(default)s)")
    parser.add_argument(
        '--alpha', type=int, default=2,
        help="Ratio of the context window for the moving-window balancing variant of parity-aware BPE (default: %(default)s)")
    parser.add_argument(
        '--num-workers', type=int, default=1,
        help="Number of processors to process texts, only supported in Python3. If -1, set `multiprocessing.cpu_count()`. (default: %(default)s)")
    parser.add_argument(
        '--apertus-formatting', action="store_true",
        help="Use Apertus-style formatting: Split(regex)+ByteLevel pre-tokenizer, BOS-only post-processor, null normalizer.")
    parser.add_argument(
        '--restart', action="store_true",
        help="Ignore any existing output file and start from scratch. Default: resume from existing output.")
    parser.add_argument(
        '--vocab-min-freq', type=int, default=2, metavar='N',
        help="Prune word types with total frequency < N before BPE learning. "
             "Dramatically reduces vocabulary size for large corpora. (default: %(default)s)")
    parser.add_argument(
        '--vocab-sample-size', type=int, default=0, metavar='N',
        help="If > 0, sample at most N word types per language (weighted by frequency) "
             "instead of using full vocabulary. Useful for very large corpora. (default: %(default)s = no sampling)")
    parser.add_argument(
        '--verbose', '-v', action="store_true",
        help="verbose mode.")

    return parser

def get_vocabulary(fobj, is_dict=False, num_workers=1, pre_tokenizer=None):
    """ Reads text and return dictionary that encodes vocabulary.
    Args:
        fobj (file-like object): The input file object to read from.
        is_dict (bool): If True, the input is treated as a dictionary file.
        num_workers (int): The number of worker processes to use for parallel processing.
        pre_tokenizer: HuggingFace pre_tokenizer to apply. Defaults to DEFAULT_PRE_TOKENIZER.
    Returns:
        Counter: A Counter object mapping tuple(word) to their frequencies.
    """
    if pre_tokenizer is None:
        pre_tokenizer = DEFAULT_PRE_TOKENIZER

    vocab = Counter()

    strip_chars = '\r\n '
    split_char = ' '

    if is_dict:
        for i, line in enumerate(fobj):
            try:
                word, count = line.strip(strip_chars).split(split_char)
            except:
                print('Failed reading vocabulary file at line {0}: {1}'.format(i, line))
                sys.exit(1)
            vocab[tuple(word)] += int(count)
    elif num_workers == 1 or fobj.name == '<stdin>':
        if num_workers > 1:
            warnings.warn("In parallel mode, the input cannot be STDIN. Using 1 processor instead.")
        for i, line in enumerate(fobj):
            # spliting the line using huggingface bpe-pre_tokenizer
            split_line = [item[0] for item in pre_tokenizer.pre_tokenize_str(line)]
            for word in split_line:
                if word:
                    vocab[tuple(word)] += 1
            
    elif num_workers > 1:

        with open_file(fobj.name, 'rb') as f: 
            size = os.fstat(f.fileno()).st_size
            chunk_size = int(size / num_workers)
            offsets = [0 for _ in range(num_workers + 1)]
            
            # Set the final offset to the end of the file
            offsets[num_workers] = size 

            for i in range(1, num_workers):
                f.seek(chunk_size * i)
                pos = f.tell()
                
                # Read to the next line break
                f.readline()
                
                offsets[i] = f.tell()
                assert 0 <= offsets[i] < 1e20, "Bad new line separator"

        pool = Pool(processes=num_workers, initializer=_init_worker, initargs=(pre_tokenizer,))
        results = []
        for i in range(num_workers):
            # Pass the file *name* and offsets to the worker
            res = pool.apply_async(_get_vocabulary, (fobj.name, offsets[i], offsets[i + 1]))
            results.append(res)
        
        pool.close()
        pool.join()

        # Collect results and sum the Counters
        for res in results:
            worker_vocab = res.get()
            vocab.update(worker_vocab)
            
    else:
        raise ValueError('`num_workers` is expected to be a positive number, but got {}.'.format(num_workers))
    return vocab

def _get_vocabulary(infile, begin, end):
    vocab = Counter()
    
    # Use the worker-level pre_tokenizer set by pool initializer
    pre_tokenizer = _worker_pre_tokenizer
    if pre_tokenizer is None:
        raise RuntimeError("_get_vocabulary called without pool initializer setting _worker_pre_tokenizer")
    
    # Open in binary mode to use the byte offsets
    with open_file(infile, 'rb') as f:
        f.seek(begin)
        
        # Read the first line (as bytes)
        line_bytes = f.readline() 
        
        while line_bytes:
            pos = f.tell()
            assert 0 <= pos < 1e20, "Bad new line separator, e.g. '\\r'"
            # Stop if we've read past the end offset
            if end > 0 and pos > end: 
                break
                
            # Decode the bytes into a string
            try:
                line_str = line_bytes.decode('utf-8', errors='strict')
            except UnicodeDecodeError as e:
                logger.warning(f"Skipping line with invalid UTF-8 at position {f.tell()}: {e}")
                line_bytes = f.readline() 
                continue 
            
            split_line = [item[0] for item in pre_tokenizer.pre_tokenize_str(line_str)]
            for word_str in split_line:
                if word_str:
                    vocab[tuple(word_str)] += 1
                        
            # Read the next line (as bytes)
            line_bytes = f.readline() 
            
    # Return the Counter directly
    return vocab


def get_vocabulary_parquet(filepath, text_column='text', num_workers=1, pre_tokenizer=None):
    """Reads a parquet file and returns a vocabulary Counter.
    
    Args:
        filepath (str): Path to the parquet file.
        text_column (str): Name of the column containing text data.
        num_workers (int): Number of worker processes (used for chunked reading).
        pre_tokenizer: HuggingFace pre_tokenizer to apply. Defaults to DEFAULT_PRE_TOKENIZER.
    
    Returns:
        Counter: A Counter object mapping tuple(word) to their frequencies.
    """
    if pre_tokenizer is None:
        pre_tokenizer = DEFAULT_PRE_TOKENIZER

    try:
        import pyarrow.parquet as pq
    except ImportError:
        logger.error("pyarrow is required to read parquet files. Install with: pip install pyarrow")
        sys.exit(1)

    vocab = Counter()
    
    pf = pq.ParquetFile(filepath)
    
    for batch in pf.iter_batches(columns=[text_column]):
        texts = batch.column(text_column).to_pylist()
        for t in texts:
            if t is None:
                continue
            for word, _span in pre_tokenizer.pre_tokenize_str(str(t)):
                if word:
                    vocab[tuple(word)] += 1

    logger.info(f"Read {sum(vocab.values())} tokens from parquet file {filepath}")
    return vocab


def pre_merge(vocab, bpe_codes):
    """Apply list of BPE merge operations to each item in vocab.
    
    Args:
        vocab (Counter): mapping from tuple(chars) -> count (tuple-keyed).
        bpe_codes (dict): mapping from (str, str) pair -> merge rank.
    
    Returns:
        Counter: new vocab with merges applied, still tuple-keyed.
    """

    new_vocab = Counter()

    for orig in vocab:

        if len(orig) == 1:
            new_vocab[orig] = vocab[orig]
            continue

        word = list(orig)

        while len(word) > 1:

            # get list of symbol pairs; optionally apply dropout
            pairs = [(bpe_codes[pair], i, pair) for (i, pair) in enumerate(zip(word, word[1:])) if pair in bpe_codes]

            if not pairs:
                break

            # get first merge operation in list of BPE codes
            bigram = min(pairs)[2]

            # find start position of all pairs that we want to merge
            positions = [i for (rank, i, pair) in pairs if pair == bigram]

            i = 0
            new_word = []
            bigram_str = ''.join(bigram)
            for j in positions:
                # merges are invalid if they start before current position. This can happen if there are overlapping pairs: (x x x -> xx x)
                if j < i:
                    continue
                new_word.extend(word[i:j]) # all symbols before merged pair
                new_word.append(bigram_str) # merged pair
                i = j + 2 # continue after merged pair
            new_word.extend(word[i:]) # add all symbols until end of word
            word = new_word

        new_vocab[tuple(word)] = vocab[orig]

    return new_vocab


def get_pair_statistics(vocab):
    """ Counts frequency of all symbol pairs, and create index.
    Args:
        vocab (list): A list of tuples, where each tuple contains a word (as a tuple of characters) and its frequency in each language.
    Returns:
        tuple: A tuple containing two dictionaries:
            - stats (defaultdict): A dictionary mapping symbol pairs (tuples) to their frequencies (numpy arrays).
            - indices (defaultdict): A dictionary mapping symbol pairs (tuples) to their indices (defaultdicts).
    """

    # data structure of pair frequencies
    stats = defaultdict(lambda: numpy.zeros(len(vocab[0][1]),dtype=int))

    #index from pairs to words
    indices = defaultdict(lambda: defaultdict(int))

    for i, (word, freq) in enumerate(vocab):
        for p in zip(word[0:-1], word[1:]):
            stats[p] += freq
            indices[p][i] += 1

    return stats, indices


def count_adjacent_pairs_tuple(word):
    """Returns dict[(sym,sym)] -> count for adjacent pairs in word tuple."""
    d = {}
    if len(word) < 2:
        return d
    prev = word[0]
    for cur in word[1:]:
        key = (prev, cur)
        d[key] = d.get(key, 0) + 1
        prev = cur
    return d

def replace_pair(pair, vocab, indices, stats, word_pair_counts):
    """
    Replaces all ('A','B') with 'AB' in vocab entries referenced by indices[pair],
    and applies stats/indices updates inline using per-word pair-count deltas.
    Returns the usual `changes` list for compatibility.
    """
    first, second = pair
    merged = first + second
    changes = []

    # Only iterate over words that actually contain the pair, per indices[pair]
    for j, occ in list(indices[pair].items()):
        if occ < 1:
            continue

        old_word, wfreq = vocab[j]
        
        # Inline merge for speed — avoid function call overhead
        n = len(old_word)
        if n < 2:
            continue
        out = []
        k = 0
        changed = False
        while k < n:
            if k + 1 < n and old_word[k] == first and old_word[k + 1] == second:
                out.append(merged)
                k += 2
                changed = True
            else:
                out.append(old_word[k])
                k += 1
        if not changed:
            continue
        new_word = tuple(out)

        # Compute pair deltas directly without building full new_pairs dict
        old_pairs = word_pair_counts[j]
        new_pairs = count_adjacent_pairs_tuple(new_word)

        # Iterate old_pairs: anything removed or changed
        for p, old_c in old_pairs.items():
            new_c = new_pairs.get(p, 0)
            d = new_c - old_c
            if d != 0:
                stats[p] += d * wfreq
                indices[p][j] += d

        # Iterate new_pairs: anything added (not in old)
        for p, new_c in new_pairs.items():
            if p not in old_pairs:
                stats[p] += new_c * wfreq
                indices[p][j] += new_c

        # commit new state
        vocab[j] = (new_word, wfreq)
        word_pair_counts[j] = new_pairs
        changes.append((j, new_word, old_word, wfreq))

    return changes


def prune_stats(stats, big_stats, threshold, full_sync=False):
    """Prunes statistics dict for efficiency of max().
    big_stats keeps full statistics for when we need to access pruned items.
    """
    if full_sync:
        # Move everything from stats to big_stats
        for item, freq in stats.items():
            if numpy.any(freq < 0):
                big_stats[item] += freq
            else:
                big_stats[item] = freq
        stats.clear()
    else:
        to_delete = []
        for item, freq in stats.items():
            if numpy.all(freq < threshold):
                to_delete.append(item)
                if numpy.any(freq < 0):
                    big_stats[item] += freq
                else:
                    big_stats[item] = freq
        for item in to_delete:
            del stats[item]


def _prune_indices(indices):
    """Remove pairs from indices whose word-count maps are all zero or empty.
    
    Without this, indices grows monotonically and holds millions of stale entries.
    """
    to_delete = []
    for pair, word_map in indices.items():
        if not word_map:
            to_delete.append(pair)
            continue
        # Remove zero-count entries within the word map
        dead_keys = [k for k, v in word_map.items() if v < 1]
        for k in dead_keys:
            del word_map[k]
        if not word_map:
            to_delete.append(pair)
    for pair in to_delete:
        del indices[pair]


def _copy_stats(source):
    """Fast copy of stats dict: shallow dict copy with numpy array .copy().
    
    Returns a defaultdict with a zero-array factory matching the source arrays,
    so new keys get auto-initialized. Much faster than copy.deepcopy.
    """
    if not source:
        return defaultdict(lambda: numpy.zeros(0, dtype=int))
    # Infer array length from first value
    sample = next(iter(source.values()))
    n = len(sample)
    result = defaultdict(lambda n=n: numpy.zeros(n, dtype=int))
    for k, v in source.items():
        result[k] = v.copy()
    return result


def _max_pair_for_index(stats, idx):
    """Find the pair with max frequency at column idx, with pair as tiebreaker.
    
    Uses batched numpy extraction when stats is large enough to benefit.
    """
    if not stats:
        return None
    n = len(stats)
    if n > 5000:
        # Batched approach: extract column into numpy array, find max, then resolve ties
        pairs = list(stats.keys())
        col = numpy.array([stats[p][idx] for p in pairs], dtype=numpy.int64)
        max_val = col.max()
        # Find all pairs with max value, pick lexicographically largest
        candidates = numpy.where(col == max_val)[0]
        if len(candidates) == 1:
            return pairs[candidates[0]]
        return max(pairs[i] for i in candidates)
    else:
        # Small dict: plain Python loop is faster
        best_pair = None
        best_freq = -1
        for pair, freq_arr in stats.items():
            f = int(freq_arr[idx])
            if f > best_freq or (f == best_freq and (best_pair is None or pair > best_pair)):
                best_freq = f
                best_pair = pair
        return best_pair


def _get_max_freq_for_index(stats, idx):
    """Return the maximum frequency value at column idx across all pairs in stats."""
    best = 0
    for freq_arr in stats.values():
        f = int(freq_arr[idx])
        if f > best:
            best = f
    return best


def _merge_tuple(word, a, b, ab):
    """Helper to merge adjacent (a, b) into ab within a word tuple."""
    n = len(word)
    if n < 2:
        return word
    out = []
    i = 0
    while i < n:
        if i + 1 < n and word[i] == a and word[i + 1] == b:
            out.append(ab)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)

def replace_pair_dict(pair, vocab, indices):
    """
    Optimized in-place merge over the dev vocab dict.
     - vocab:   defaultdict[tuple -> np.ndarray]
     - indices: defaultdict[pair -> set[tuple]]
    """
    first, second = pair
    merged = first + second
    length_change = None

    # Iterate over a static list since we modify indices during the loop
    words_to_process = list(indices[pair])

    for old_word in words_to_process:
        
        # Word may have been deleted by a previous merge in this same loop
        if old_word not in vocab:
            continue
            
        freq = vocab[old_word]
        new_word = _merge_tuple(old_word, first, second, merged)

        if new_word == old_word:
            continue
            
        del vocab[old_word]
        vocab[new_word] += freq  

        # Track length change
        if length_change is None:
            length_change = np.zeros(len(freq), dtype=int)
        length_change += (len(old_word) - len(new_word)) * freq

        # Update indices map: remove old_word from all pair indices
        old_pairs = count_adjacent_pairs_tuple(old_word)
        for p in old_pairs:
            if p in indices and old_word in indices[p]:
                indices[p].remove(old_word)
                if not indices[p]:
                    del indices[p]

        # Add new_word to all pair indices
        new_pairs = count_adjacent_pairs_tuple(new_word)
        for p in new_pairs:
            indices[p].add(new_word)

    if length_change is None:
        length_change = 0

    return length_change


@contextmanager
def open_file(filename, mode):
    if mode in ('r', 'w'):
        f = open(filename, mode, encoding="utf-8")
    elif mode in ('rb', 'wb'):
        f = open(filename, mode)
    try:
        yield f
    finally:
        f.close()


def preprocess_input_data(infiles, devfiles, is_dict=False, total_symbols=False, num_global=0, num_workers=1, bpe_file=None, text_column='text', vocab_min_freq=2, vocab_sample_size=0, pre_tokenizer=None):
    """ Reads input files and creates vocabulary data structure.
    Args:
        infiles (list[str]): A list of input file paths.
        devfiles (list[str]): A list of development file paths.
        is_dict (bool): Whether the input files are in dictionary format.
        total_symbols (bool): Whether to count total symbols.
        num_global (int): The number of global symbols.
        num_workers (int): The number of worker threads to use.
        bpe_file (fobj): file containing merge operations to pre-apply before learning.
        pre_tokenizer: HuggingFace pre_tokenizer to apply. Defaults to DEFAULT_PRE_TOKENIZER.
    Returns:
        tuple: A tuple containing:
            - dev_vocab (defaultdict): A dictionary mapping subwords to their frequencies in the development set.
            - dev_indices (defaultdict): A dictionary mapping pairs to sets of words in dev_vocab.
            - sorted_vocab (list): A sorted list of tuples, where each tuple contains a subword and its frequency in each language.
            - stats (defaultdict): A dictionary mapping symbol pairs (tuples) to their frequencies (numpy arrays).
            - word_pair_counts (list): Per-word adjacent pair count dicts for sorted_vocab.
            - indices (defaultdict): A dictionary mapping symbol pairs (tuples) to their indices (defaultdicts).
            - big_stats (defaultdict): A dictionary containing full statistics for all symbol pairs.
            - threshold (numpy.ndarray): An array of thresholds for pruning statistics.
            - lengths (numpy.ndarray or None): An array where each element corresponds to the sum of frequency*length for all vocab items in the development set.
            - array_length (int): The length of the vocabulary array, which is the number of languages plus one for concatenation.
    """
    if pre_tokenizer is None:
        pre_tokenizer = DEFAULT_PRE_TOKENIZER

    _log_memory("start preprocessing")

    if bpe_file is not None:
        # ignore first line containing version information (if it exists)
        line = bpe_file.readline()
        offset = 1
        if not line.startswith('version'):
            bpe_file.seek(0)
            offset = 0
        
        bpe_codes = [tuple(item.strip('\r\n ').split(' ')) for (n, item) in enumerate(bpe_file.read().rstrip('\n').split('\n'))]

        for i, item in enumerate(bpe_codes):
            if len(item) != 2:
                sys.stderr.write('Error: invalid line {0} in BPE codes file: {1}\n'.format(i+offset, ' '.join(item)))
                sys.stderr.write('The line should exist of exactly two subword units, separated by whitespace\n')
                sys.exit(1)

        # some hacking to deal with duplicates (only consider first instance)
        bpe_codes = dict([(code, i) for (i, code) in reversed(list(enumerate(bpe_codes)))])
    else:
        bpe_codes = None

    vocabs = []
    joint_keys = set()
    for lang_i, f in enumerate(infiles):
        # f can be a single file/path or a list of files/paths (multi-file per language)
        if isinstance(f, list):
            file_list = f
        else:
            file_list = [f]

        # Resolve per-language text column
        if isinstance(text_column, list):
            col = text_column[lang_i]
        else:
            col = text_column

        lang_vocab = Counter()
        for ff in file_list:
            fname = ff if isinstance(ff, str) else getattr(ff, 'name', '')
            if fname.endswith('.parquet'):
                v = get_vocabulary_parquet(fname, text_column=col, num_workers=num_workers, pre_tokenizer=pre_tokenizer)
            else:
                v = get_vocabulary(ff, is_dict, num_workers, pre_tokenizer=pre_tokenizer)
            lang_vocab.update(v)

        if bpe_codes is not None:
            lang_vocab = pre_merge(lang_vocab, bpe_codes)
        vocabs.append(lang_vocab)
        joint_keys = joint_keys.union(lang_vocab.keys())
        _log_memory(f"after loading language {lang_i}")

    # --- Vocabulary pruning and sampling ---
    total_types_before = sum(len(v) for v in vocabs)
    total_tokens_before = sum(sum(v.values()) for v in vocabs)

    if vocab_sample_size > 0:
        # Weighted sampling: keep top-N word types per language by frequency
        import random
        for lang_i in range(len(vocabs)):
            v = vocabs[lang_i]
            if len(v) <= vocab_sample_size:
                continue
            # Keep the top vocab_sample_size types by frequency
            top_items = v.most_common(vocab_sample_size)
            vocabs[lang_i] = Counter(dict(top_items))
        # Rebuild joint_keys after sampling
        joint_keys = set()
        for v in vocabs:
            joint_keys = joint_keys.union(v.keys())

    if vocab_min_freq > 1:
        # Prune word types whose total frequency across all languages is below threshold
        # First, compute total frequency per word type
        total_freq = Counter()
        for v in vocabs:
            total_freq.update(v)
        # Keep only types with total freq >= vocab_min_freq
        keep_keys = {k for k, f in total_freq.items() if f >= vocab_min_freq}
        for lang_i in range(len(vocabs)):
            vocabs[lang_i] = Counter({k: c for k, c in vocabs[lang_i].items() if k in keep_keys})
        joint_keys = keep_keys
        del total_freq, keep_keys

    total_types_after = sum(len(v) for v in vocabs)
    total_tokens_after = sum(sum(v.values()) for v in vocabs)
    if total_types_after < total_types_before:
        logger.info(f"Vocabulary reduction: {total_types_before:,} → {total_types_after:,} word types "
                     f"({100*total_types_after/total_types_before:.1f}%), "
                     f"{total_tokens_before:,} → {total_tokens_after:,} tokens "
                     f"({100*total_tokens_after/total_tokens_before:.1f}%)")

    dev_vocabs = []
    dev_keys = set()
    if devfiles:
        for f in devfiles:
            vocab = get_vocabulary(f, is_dict, num_workers, pre_tokenizer=pre_tokenizer)
            if bpe_codes is not None:
                vocab = pre_merge(vocab, bpe_codes)
            dev_vocabs.append(vocab)
            dev_keys = dev_keys.union(vocab.keys())

    array_length = len(vocabs)
    if num_global:
        array_length += 1

    # merge vocabularies. Data structure maps from subword to list of frequency in each language, plus one for concatenation
    vocab = defaultdict(lambda: numpy.zeros(array_length, dtype=int))
    for i, v in enumerate(vocabs):
        for key, cnt in v.items():
            vocab[key][i] = cnt

    if num_global:
        for key in joint_keys:
            vocab[key][-1] = sum(vocab[key])

    # Free per-language vocabs and joint_keys — their data is now in the joint vocab
    del vocabs, joint_keys
    gc.collect()
    _log_memory("after merging vocabs")

    # merge dev vocabularies. Data structure maps from subword to list of frequency in each language
    dev_vocab = defaultdict(lambda: numpy.zeros(len(dev_vocabs), dtype=int))
    
    if dev_vocabs:
        for i, v in enumerate(dev_vocabs):
            for key, cnt in v.items():
                dev_vocab[key][i] = cnt

    dev_indices = defaultdict(set)
    for word, freq in dev_vocab.items():
        for p in zip(word[0:-1], word[1:]):
            dev_indices[p].add(word)

    sorted_vocab = sorted(vocab.items(), key=lambda x: sum(x[1]), reverse=True)

    # Free the merged vocab dict — sorted_vocab now owns the data
    del vocab

    stats, indices = get_pair_statistics(sorted_vocab)
    # big_stats starts EMPTY: only pruned entries are stored here.
    # This avoids duplicating all pair statistics (~50% memory savings on stats).
    # The positive/negative heuristic in prune_stats correctly establishes baselines
    # on first prune, and periodic full_sync+swap corrects any accumulated drift.
    big_stats = defaultdict(lambda: numpy.zeros(array_length, dtype=int))

    word_pair_counts = [count_adjacent_pairs_tuple(word) for (word, _freq) in sorted_vocab]

    _log_memory("after pair statistics")
    logger.info(f"Stats size: {len(stats):,} pairs, sorted_vocab size: {len(sorted_vocab):,} word types")

    if total_symbols:
        uniq_char_internal = set()
        uniq_char_final = set()
        for word, _freq in sorted_vocab:
            for char in word[:-1]:
                uniq_char_internal.add(char)
            uniq_char_final.add(word[-1])
        sys.stderr.write('Number of word-internal characters: {0}\n'.format(len(uniq_char_internal)))
        sys.stderr.write('Number of word-final characters: {0}\n'.format(len(uniq_char_final)))
        sys.stderr.write('Reducing number of merge operations by {0}\n'.format(len(uniq_char_internal) + len(uniq_char_final)))
        # num_symbols -= len(uniq_char_internal) + len(uniq_char_final)

    # threshold is inspired by Zipfian assumption, but should only affect how often we re-sync with full big_stats
    threshold = numpy.zeros(array_length, dtype=float)
    for l in range(array_length):
        threshold[l] = _get_max_freq_for_index(stats, l) / 10

    if dev_vocab:
        lengths = functools.reduce(numpy.add, [len(key)*value for key, value in dev_vocab.items()])
    else:
        lengths = None
    return (dev_vocab, dev_indices, sorted_vocab, stats, word_pair_counts, indices, big_stats, threshold, lengths, array_length)


def _read_merge_lines(filepath):
    """Read merge lines from an existing BPE output file."""
    if not os.path.exists(filepath):
        return []
    lines = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip('\r\n ')
            if not line or line.startswith('#version') or line.startswith('version'):
                continue
            parts = line.split(' ')
            if len(parts) == 2:
                lines.append(line)
    return lines


def _make_bpe_file_from_lines(merge_lines):
    """Create a file-like object from merge lines for use as bpe_file in preprocess_input_data."""
    if not merge_lines:
        return None
    content = '#version: 0.2\n' + '\n'.join(merge_lines) + '\n'
    return io.StringIO(content)


def _replay_merges_to_hf_vocab(merge_lines, vocab, merges):
    """Replay merge lines into the HF vocab dict and merges list."""
    for line in merge_lines:
        parts = line.strip().split(' ')
        if len(parts) != 2:
            continue
        s1, s2 = parts
        merged_token = s1 + s2
        merges.append((s1, s2))
        if merged_token not in vocab:
            vocab[merged_token] = len(vocab)


def save_tokenizer_json(path, vocab, merges, training_pre_tokenizer=None, uses_bytelevel=False, apertus_formatting=False, unk_token="<unk>", special_tokens=None):
    """Saves the learned vocab and merges to a HF tokenizer.json.
    
    The pre_tokenizer saved to the file matches exactly what was used during training.
    A ByteLevel decoder is set only if ByteLevel pre-tokenization was applied.
    
    Args:
        path (str): The file path to save to.
        vocab (dict): The complete vocabulary (str -> int).
        merges (list): The list of merge rules as (str, str) tuples.
        training_pre_tokenizer: The HF pre_tokenizer object used during training.
        uses_bytelevel (bool): Whether ByteLevel pre-tokenization was used during training.
        apertus_formatting (bool): If True, use Apertus-style post-processor (BOS-only, null normalizer).
        unk_token (str): The unknown token.
        special_tokens (list): List of special token strings.
    """
    if special_tokens is None:
        special_tokens = SPECIAL_TOKENS

    bpe_model = models.BPE(vocab=vocab, merges=merges, unk_token=unk_token)
    tokenizer = Tokenizer(bpe_model)
    tokenizer.add_special_tokens(special_tokens)

    # Set pre-tokenizer to exactly what was used during training
    tokenizer.pre_tokenizer = training_pre_tokenizer

    # Set decoder only if ByteLevel pre-tokenization was used
    if uses_bytelevel:
        tokenizer.decoder = decoders.ByteLevel()

    bos_id = vocab.get("<s>", 0)
    eos_id = vocab.get("</s>", 2)

    if apertus_formatting:
        tokenizer.normalizer = None
        tokenizer.post_processor = TemplateProcessing(
            single="<s> $A",
            pair="<s> $A <s>:1 $B:1",
            special_tokens=[("<s>", bos_id)]
        )
    else:
        tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> </s> $B </s>",
            special_tokens=[("<s>", bos_id), ("</s>", eos_id)]
        )

    try:
        tokenizer.save(path)
        logger.info(f"Successfully saved Hugging Face tokenizer to {path}")
    except Exception as e:
        logger.error(f"Failed to save tokenizer.json to {path}: {e}")


def learn_bpe(infiles, outfile, devfiles, num_symbols, min_frequency=2, verbose=False, is_dict=False, total_symbols=False, num_global=0, ratio=None, num_workers=1, bpe_file=None, text_column='text', resume_lines=None, vocab_min_freq=2, vocab_sample_size=0, pre_tokenizer=None):
    """Learn num_symbols merge operations using Parity-aware BPE."""
    if pre_tokenizer is None:
        pre_tokenizer = DEFAULT_PRE_TOKENIZER
    logger.info("Learning parity-aware BPE with the following parameters:"
          "\n  num_symbols: {0}, min_frequency: {1}, verbose: {2}, is_dict: {3}, total_symbols: {4}, num_global: {5}, num_workers: {6}".format(
              num_symbols, min_frequency, verbose, is_dict, total_symbols, num_global, num_workers))
    
    # --- HF Tokenizer Init ---
    initial_alphabet = pre_tokenizers.ByteLevel.alphabet()
    vocab = {token: i for i, token in enumerate(SPECIAL_TOKENS + initial_alphabet)}
    merges = []

    # Combine preload + resume merges
    preload_lines = []
    if bpe_file is not None:
        bpe_file.seek(0)
        for line in bpe_file:
            line = line.strip('\r\n ')
            if not line or line.startswith('#version') or line.startswith('version'):
                continue
            if len(line.split(' ')) == 2:
                preload_lines.append(line)

    if resume_lines is None:
        resume_lines = []

    all_prior_lines = preload_lines + resume_lines
    num_completed = len(resume_lines)

    _replay_merges_to_hf_vocab(all_prior_lines, vocab, merges)
    combined_bpe_file = _make_bpe_file_from_lines(all_prior_lines)

    if num_completed == 0:
        outfile.write('#version: 0.2\n')
        for line in preload_lines:
            outfile.write(line + '\n')

    if num_completed > 0:
        logger.info(f"Resuming from {num_completed} previously learned merges")

    remaining = num_symbols - num_completed
    if remaining <= 0:
        logger.info(f"Already have {num_completed} merges >= {num_symbols} requested. Nothing to do.")
        return vocab, merges

    dev_vocab, dev_indices, sorted_vocab, stats, word_pair_counts, indices, big_stats, threshold, lengths, array_length = \
        preprocess_input_data(infiles, devfiles, is_dict, total_symbols, num_global, num_workers, combined_bpe_file, text_column, vocab_min_freq=vocab_min_freq, vocab_sample_size=vocab_sample_size, pre_tokenizer=pre_tokenizer)

    if ratio is not None:
        initial_lengths = functools.reduce(numpy.add, [len(key)*value for key, value in sorted_vocab])
        lengths = numpy.copy(initial_lengths)

    for i in tqdm(range(remaining), desc="parity-aware BPE..."):
        global_i = i + num_completed

        if stats:
            if global_i < num_global:
                logger.info('lengths {0}: picking best subword based on concatenation'.format(lengths))
                max_index = -1
            else:
                if ratio is not None:
                    compression_rates = initial_lengths / lengths
                    adjusted_compression_rates = compression_rates / ratio
                    max_index, max_value = min(enumerate(adjusted_compression_rates), key=operator.itemgetter(1))
                    logger.info('initial lengths {0}\nlengths {1}'.format(initial_lengths, lengths))
                    logger.info('compression rates {0}\nadjusted compression rates {1}: picking best subword in corpus {2}'.format(
                        compression_rates, adjusted_compression_rates, max_index))
                else:
                    max_index, max_value = max(enumerate(lengths), key=operator.itemgetter(1))
                    logger.info('lengths {0}: picking best subword in corpus {1}'.format(lengths, max_index))

            most_frequent = _max_pair_for_index(stats, max_index)

        if not stats or (i and stats[most_frequent][max_index] < threshold[max_index]):
            prune_stats(stats, big_stats, threshold, full_sync=True)
            # Swap instead of _copy_stats: avoids allocating a full third copy at peak
            # After full_sync, stats is empty and big_stats has all corrected values.
            # Swapping gives stats the corrected values and big_stats an empty dict.
            stats, big_stats = big_stats, stats
            most_frequent = _max_pair_for_index(stats, max_index)
            for l in range(array_length):
                threshold[l] = _get_max_freq_for_index(stats, l) * global_i/(global_i+10000.0)
            prune_stats(stats, big_stats, threshold)

        if stats[most_frequent][max_index] < min_frequency:
            sys.stderr.write(f'no pair has frequency >= {min_frequency}. Stopping for language {max_index} with length: {lengths}\n')
            break

        logger.info('pair {0}: {1} {2} -> {1}{2} (frequency {3})'.format(global_i, most_frequent[0], most_frequent[1], stats[most_frequent]))

        s1, s2 = most_frequent
        merged_token = s1 + s2
        merges.append((s1, s2))
        if merged_token not in vocab:
            vocab[merged_token] = len(vocab)
        
        outfile.write(f"{s1} {s2}\n")
        outfile.flush()
        
        changes = replace_pair(most_frequent, sorted_vocab, indices, stats, word_pair_counts)

        if ratio is not None:
            length_change = numpy.zeros(array_length, dtype=int)
            for _j, new_w, old_w, wfreq in changes:
                length_change += (len(old_w) - len(new_w)) * wfreq
            lengths -= length_change
        else:
            length_change = replace_pair_dict(most_frequent, dev_vocab, dev_indices)
            lengths -= length_change
        
        if not i % 100:
            prune_stats(stats, big_stats, threshold)
            _prune_indices(indices)

        if not i % 1000:
            _log_memory(f"merge {global_i}")
            logger.info(f"  stats: {len(stats):,}, big_stats: {len(big_stats):,}, indices: {len(indices):,}")

        stats[most_frequent] = numpy.zeros(array_length, dtype=int)
        indices[most_frequent] = defaultdict(int)

    return vocab, merges


def select_language_index(lengths, selected_indices, selection_threshold, window_size):
    """ Selects the index of the language with the maximum length from the remaining valid indices.
    The selection is based on a moving window approach, where indices that have been selected too often
    are excluded from further consideration.

    Args:
        lengths (numpy.ndarray): An array of lengths for each language.
        selected_indices (deque): A deque containing the indices of previously selected languages.
        selection_threshold (float): The threshold ratio for selecting an index.
        window_size (int): The size of the moving window.

    Returns:
        int: The index of the selected language.
    """
    final_index = -1
    # Boolean mask to keep track of valid indices
    mask = numpy.ones(len(lengths), dtype=bool)  # Start with all elements valid

    while True:
        # Find the maximum index in the remaining elements
        valid_indices = numpy.where(mask)[0]  # Indices of unmasked elements
        max_index = valid_indices[numpy.argmax(lengths[valid_indices])]  # Max in valid range
        count = selected_indices.count(max_index)
        ratio = count * 1.0 / window_size
        if ratio <= selection_threshold:
            final_index = max_index
            break
        else:
            # Exclude this index from further consideration
            mask[max_index] = False

    return final_index

def learn_bpe_moving_window(infiles, outfile, devfiles, num_symbols, window_size=100, alpha=2, min_frequency=2, verbose=False, is_dict=False, total_symbols=False, num_global=0, ratio=None, num_workers=1, bpe_file=None, text_column='text', resume_lines=None, vocab_min_freq=2, vocab_sample_size=0, pre_tokenizer=None):
    """Learn num_symbols merge operations using Parity-aware BPE (moving-window variant)."""
    if pre_tokenizer is None:
        pre_tokenizer = DEFAULT_PRE_TOKENIZER
    logger.info("Using Parity-aware BPE (moving-window variant) with window size {0} and alpha {1}".format(window_size, alpha))
    logger.info("Learning parity-aware BPE with the following parameters:"
          "\n  num_symbols: {0}, min_frequency: {1}, verbose: {2}, is_dict: {3}, total_symbols: {4}, num_global: {5}, num_workers: {6}".format(
              num_symbols, min_frequency, verbose, is_dict, total_symbols, num_global, num_workers))
    
    # --- HF Tokenizer Init ---
    initial_alphabet = pre_tokenizers.ByteLevel.alphabet()
    vocab = {token: i for i, token in enumerate(SPECIAL_TOKENS + initial_alphabet)}
    merges = []

    # Combine preload + resume merges
    preload_lines = []
    if bpe_file is not None:
        bpe_file.seek(0)
        for line in bpe_file:
            line = line.strip('\r\n ')
            if not line or line.startswith('#version') or line.startswith('version'):
                continue
            if len(line.split(' ')) == 2:
                preload_lines.append(line)

    if resume_lines is None:
        resume_lines = []

    all_prior_lines = preload_lines + resume_lines
    num_completed = len(resume_lines)

    _replay_merges_to_hf_vocab(all_prior_lines, vocab, merges)
    combined_bpe_file = _make_bpe_file_from_lines(all_prior_lines)

    if num_completed == 0:
        if preload_lines:
            outfile.write('#version: 0.2\n')
            for line in preload_lines:
                outfile.write(line + '\n')
        else:
            outfile.write('#version: 0.2\n')

    if num_completed > 0:
        logger.info(f"Resuming from {num_completed} previously learned merges")

    remaining = num_symbols - num_completed
    if remaining <= 0:
        logger.info(f"Already have {num_completed} merges >= {num_symbols} requested. Nothing to do.")
        return vocab, merges

    dev_vocab, dev_indices, sorted_vocab, stats, word_pair_counts, indices, big_stats, threshold, lengths, array_length = \
        preprocess_input_data(infiles, devfiles, is_dict, total_symbols, num_global, num_workers, combined_bpe_file, text_column, vocab_min_freq=vocab_min_freq, vocab_sample_size=vocab_sample_size, pre_tokenizer=pre_tokenizer)

    if ratio is not None:
        initial_lengths = functools.reduce(numpy.add, [len(key)*value for key, value in sorted_vocab])
        lengths = numpy.copy(initial_lengths)

    # Use the number of languages (not array_length, which includes the concatenation slot when num_global > 0)
    num_languages = len(infiles)
    selection_threshold = alpha * 1.0 / num_languages
    selected_indices = deque(maxlen=window_size)

    for i in tqdm(range(remaining), desc="Parity-aware BPE (moving-window variant)... \n"):
        global_i = i + num_completed

        if stats:
            if global_i < num_global:
                logger.info('lengths {0}: picking best subword based on concatenation'.format(lengths))
                max_index = -1
            else:
                if ratio is not None:
                    compression_rates = initial_lengths / lengths
                    adjusted_compression_rates = compression_rates / ratio
                    max_index = select_language_index(-adjusted_compression_rates, selected_indices, selection_threshold, window_size)
                    selected_indices.append(max_index)
                    logger.info('initial lengths {0}\nlengths {1}'.format(initial_lengths, lengths))
                    logger.info('compression rates {0}\nadjusted compression rates {1}: picking best subword in corpus {2}'.format(
                        compression_rates, adjusted_compression_rates, max_index))
                else:
                    max_index = select_language_index(lengths, selected_indices, selection_threshold, window_size)
                    selected_indices.append(max_index)
                    logger.info('lengths {0}: picking best subword in corpus {1}'.format(lengths, max_index))

            most_frequent = _max_pair_for_index(stats, max_index)

        if not stats or (i and stats[most_frequent][max_index] < threshold[max_index]):
            prune_stats(stats, big_stats, threshold, full_sync=True)
            # Swap instead of _copy_stats: avoids allocating a full third copy at peak
            stats, big_stats = big_stats, stats
            most_frequent = _max_pair_for_index(stats, max_index)
            for l in range(array_length):
                threshold[l] = _get_max_freq_for_index(stats, l) * global_i/(global_i+10000.0)
            prune_stats(stats, big_stats, threshold)

        if stats[most_frequent][max_index] < min_frequency:
            sys.stderr.write(f'no pair has frequency >= {min_frequency}. Stopping for language {max_index} with length: {lengths}\n')
            break

        logger.info('pair {0}: {1} {2} -> {1}{2} (frequency {3})'.format(global_i, most_frequent[0], most_frequent[1], stats[most_frequent]))

        s1, s2 = most_frequent
        merged_token = s1 + s2
        merges.append((s1, s2))
        if merged_token not in vocab:
            vocab[merged_token] = len(vocab)
        
        outfile.write(f"{s1} {s2}\n")
        outfile.flush()

        changes = replace_pair(most_frequent, sorted_vocab, indices, stats, word_pair_counts)

        if ratio is not None:
            length_change = numpy.zeros(array_length, dtype=int)
            for _j, new_w, old_w, wfreq in changes:
                length_change += (len(old_w) - len(new_w)) * wfreq
            lengths -= length_change
        else:
            length_change = replace_pair_dict(most_frequent, dev_vocab, dev_indices)
            lengths -= length_change
        
        if not i % 100:
            prune_stats(stats, big_stats, threshold)
            _prune_indices(indices)

        if not i % 1000:
            _log_memory(f"merge {global_i}")
            logger.info(f"  stats: {len(stats):,}, big_stats: {len(big_stats):,}, indices: {len(indices):,}")

        stats[most_frequent] = numpy.zeros(array_length, dtype=int)
        indices[most_frequent] = defaultdict(int)
    
    return vocab, merges

if __name__ == '__main__':

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    newdir = os.path.join(currentdir, 'subword_nmt')
    if os.path.isdir(newdir):
        warnings.warn(
            "this script's location has moved to {0}. This symbolic link will be removed in a future version. Please point to the new location, or install the package and use the command 'subword-nmt'".format(newdir),
            DeprecationWarning
        )

    parser = create_parser()
    args = parser.parse_args()

    sys.stderr = codecs.getwriter('UTF-8')(sys.stderr.buffer)
    sys.stdout = codecs.getwriter('UTF-8')(sys.stdout.buffer)
    sys.stdin = codecs.getreader('UTF-8')(sys.stdin.buffer)

    set_logger(args.verbose)

    if args.num_workers <= 0:
        args.num_workers = cpu_count() - 1
    
    if sys.version_info < (3, 0):
        print("Python 2 is deprecated. Use Python 3")
        sys.exit(1)

    
    # --- Config file handling ---
    if args.config:
        with open(args.config, 'r') as cf:
            config = json.load(cf)
        languages = config['languages']
        args.input = []
        text_columns = []
        for lang in languages:
            paths = lang['input']
            # Support both single path string and list of paths
            if isinstance(paths, str):
                paths = [paths]
            lang_files = []
            for path in paths:
                if path.endswith('.parquet'):
                    lang_files.append(path)
                else:
                    lang_files.append(codecs.open(path, encoding='utf-8'))
            # Store as list (even if single file) for uniform handling
            args.input.append(lang_files if len(lang_files) > 1 else lang_files[0])
            # Per-language text column, falling back to CLI default
            text_columns.append(lang.get('text_column', args.text_column))
        args.text_column = text_columns
        if any('dev' in lang for lang in languages):
            args.dev = [codecs.open(lang['dev'], encoding='utf-8') for lang in languages]
        if any('ratio' in lang for lang in languages):
            args.ratio = numpy.array([lang['ratio'] for lang in languages])
            args.ratio = args.ratio / args.ratio[0]

    # --- Validation ---
    if args.dev:
        assert len(args.input) == len(args.dev)

    if args.ratio is not None:
        assert args.dev is None
        assert len(args.input) == len(args.ratio)
        if not isinstance(args.ratio, numpy.ndarray):
            args.ratio = numpy.array(args.ratio)
            args.ratio = args.ratio / args.ratio[0]

    if args.dev is None and args.ratio is None:
        print("script requires either dev sets or ratios")
        sys.exit(1)

    # --- Open input files ---
    if not args.config:
        if args.input is None:
            args.input = [sys.stdin]
        else:
            for i, f in enumerate(args.input):
                if isinstance(f, str) and not f.endswith('.parquet'):
                    args.input[i] = codecs.open(f, encoding='utf-8')
        if args.dev:
            for i, f in enumerate(args.dev):
                args.dev[i] = codecs.open(f.name, encoding='utf-8')

    # --- Pre-emption: check for existing output ---
    resume_lines = []
    output_path = args.output

    if output_path and not args.restart and os.path.exists(output_path):
        resume_lines = _read_merge_lines(output_path)
        if resume_lines:
            logger.info(f"Found existing output with {len(resume_lines)} merges. Resuming. Use --restart to start fresh.")

    # --- Open output file ---
    if output_path is None:
        outfile = sys.stdout
    elif resume_lines:
        outfile = open(output_path, 'a', encoding='utf-8')
    else:
        outfile = open(output_path, 'w', encoding='utf-8')

    # --- Preload BPE file ---
    if args.preload is None:
        bpe_file = None
    else:
        bpe_file = codecs.open(args.preload.name, encoding='utf-8')

    # --- Configure pre-tokenizer ---
    uses_bytelevel = False
    if args.apertus_formatting:
        uses_bytelevel = True
        pre_tokenizer = pre_tokenizers.Sequence([
            Split(pattern=Regex(APERTUS_V2_SPLIT_PATTERN), behavior='isolated', invert=False),
            ByteLevel(use_regex=False, add_prefix_space=False)
        ])
    else:
        pretokenizer_list = []
        for pt in args.pretokenize:
            if pt == 'whitespace':
                pretokenizer_list.append(Whitespace())
            elif pt == 'bytelevel':
                uses_bytelevel = True
                pretokenizer_list.append(ByteLevel(use_regex=False, add_prefix_space=True))
            else:
                raise ValueError("pretokenizer {0} is not implemented".format(pt))
        if not pretokenizer_list:
            raise ValueError("--pretokenize requires at least one pretokenizer (got empty list). "
                             "Use e.g. --pretokenize whitespace bytelevel")
        pre_tokenizer = pre_tokenizers.Sequence(pretokenizer_list)

    # --- Run training ---
    if args.variant == 'base':
        vocab, merges = learn_bpe(args.input, outfile, args.dev, args.symbols, args.min_frequency, args.verbose,
            num_global=args.global_merges, is_dict=args.dict_input, total_symbols=args.total_symbols,
            ratio=args.ratio, num_workers=args.num_workers, bpe_file=bpe_file, text_column=args.text_column,
            resume_lines=resume_lines if resume_lines else None,
            vocab_min_freq=args.vocab_min_freq, vocab_sample_size=args.vocab_sample_size,
            pre_tokenizer=pre_tokenizer)
    elif args.variant == 'window':
        vocab, merges = learn_bpe_moving_window(args.input, outfile, args.dev, args.symbols, args.window_size,
            args.alpha, args.min_frequency, args.verbose, num_global=args.global_merges, is_dict=args.dict_input,
            total_symbols=args.total_symbols, ratio=args.ratio, num_workers=args.num_workers, bpe_file=bpe_file,
            text_column=args.text_column, resume_lines=resume_lines if resume_lines else None,
            vocab_min_freq=args.vocab_min_freq, vocab_sample_size=args.vocab_sample_size,
            pre_tokenizer=pre_tokenizer)
    else:
        raise ValueError("Unknown BPE variant: {0}. Use 'base' or 'window'.".format(args.variant))

    if args.json_output and vocab and merges:
        save_tokenizer_json(args.json_output, vocab, merges,
            training_pre_tokenizer=pre_tokenizer, uses_bytelevel=uses_bytelevel,
            apertus_formatting=args.apertus_formatting)

    # --- Cleanup ---
    def _close_file(f):
        if isinstance(f, str):
            return
        if isinstance(f, list):
            for ff in f:
                _close_file(ff)
            return
        if hasattr(f, 'name') and f.name != '<stdin>':
            f.close()

    for f in args.input:
        _close_file(f)
    if output_path:
        outfile.close()
