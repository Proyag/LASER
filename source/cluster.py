import os
import tempfile
import sys
import time
import argparse
import numpy as np
import faiss

# get environment
assert os.environ.get('LASER'), 'Please set the environment variable LASER'
LASER = os.environ['LASER']

sys.path.append(LASER + '/source/lib')
from text_processing import Token, BPEfastApply
from embed import SentenceEncoder, EncodeTime, buffered_read

LASER_DIM = 1024


# Write clustered sentences to their corresponding output files
def WriteBatch(sentences, raw_sentences, cluster_ids, out_files):
    counts = np.zeros(len(out_files), dtype=int)
    cluster_ids = cluster_ids.flatten().tolist()
    assert len(raw_sentences) == len(sentences) == len(cluster_ids), 'Incorrect number of cluster ids/sentences'
    for sentence, raw_sentence, cluster_id in zip(sentences, raw_sentences, cluster_ids):
        out_files[cluster_id].write(raw_sentence + '\n')
        counts[cluster_id] += 1
    return counts


# Cluster sentences (existing file pointers)
def ClusterFilep(encoder, raw_inp_file, bpe_inp_file, out_files,
                 buffer_size=10000, verbose=False,
                 num_clusters=10, niter=25, nredo=1,
                 min_points_per_centroid=39, max_points_per_centroid=256,
                 spherical=False, update_index=False, gpu_kmeans=False):
    n = 0
    cluster_counts = np.zeros(len(out_files), dtype=int)
    t = time.time()
    kmeans = faiss.Kmeans(LASER_DIM, num_clusters,
                          niter=niter, nredo=nredo,
                          min_points_per_centroid=min_points_per_centroid,
                          max_points_per_centroid=max_points_per_centroid,
                          verbose=verbose, spherical=spherical,
                          update_index=update_index, gpu=gpu_kmeans)
    encoded = np.empty((0, LASER_DIM), dtype=np.float32)
    for (sentences, raw_sentences) in zip(buffered_read(bpe_inp_file, buffer_size),
                                          buffered_read(raw_inp_file, buffer_size)):
        n += len(sentences)
        if not hasattr(kmeans, 'index') or not kmeans.index.is_trained:
            encoded = np.vstack((encoded, encoder.encode_sentences(sentences)))
            if n >= max_points_per_centroid * num_clusters:
                # Train Kmeans
                kmeans.train(encoded)
                _, cluster_ids = kmeans.index.search(encoded, 1)
                cluster_counts += WriteBatch(sentences, raw_sentences, cluster_ids, out_files)
        else:
            _, cluster_ids = kmeans.index.search(encoder.encode_sentences(sentences), 1)
            cluster_counts += WriteBatch(sentences, raw_sentences, cluster_ids, out_files)
        if verbose and n % 10000 == 0:
            print('\r - Clustering: {:d} sentences'.format(n), end='')
    if verbose:
        print('\r - Clustering: Clustered {:d} sentences'.format(n), end='')
        EncodeTime(t)            


# Cluster sentences (file names)
def ClusterFile(encoder, raw_fname, bpe_fname, out_prefix, inp_encoding='utf-8',
                buffer_size=10000, verbose=False,
                num_clusters=10, niter=25, nredo=1,
                min_points_per_centroid=39, max_points_per_centroid=256,
                spherical=False, update_index=False, gpu_kmeans=False):
    if verbose:
        print(' - Encoder: Clustering {} to {}'.
              format(os.path.basename(bpe_fname) if len(bpe_fname) > 0 else 'stdin',
                     os.path.basename(out_prefix) + '.*'))
    fin_raw = open(raw_fname, 'r', encoding=inp_encoding, errors='surrogateescape')
    fin_bpe = open(bpe_fname, 'r', encoding=inp_encoding, errors='surrogateescape')
    fouts = [open(out_prefix + '.cluster_{}'.format(i), mode='w') for i in range(num_clusters)]
    ClusterFilep(encoder, fin_raw, fin_bpe, fouts,
                 buffer_size=buffer_size, verbose=verbose,
                 num_clusters=num_clusters, niter=niter, nredo=nredo,
                 min_points_per_centroid=min_points_per_centroid,
                 max_points_per_centroid=max_points_per_centroid,
                 spherical=spherical, update_index=update_index, gpu_kmeans=gpu_kmeans)
    fin_raw.close()
    fin_bpe.close()
    for fout in fouts:
        fout.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LASER: Embed sentences and cluster them')
    parser.add_argument('--encoder', type=str, required=True,
                        help='encoder to be used')
    parser.add_argument('--token-lang', type=str, default='--',
                        help="Perform tokenization with given language ('--' for no tokenization)")
    parser.add_argument('--bpe-codes', type=str, default=None,
                        help='Apply BPE using specified codes')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Detailed output')

    parser.add_argument('-i', '--input', required=True,
                        help='Input text file')
    parser.add_argument('-o', '--output', required=True,
                        help='Output file prefix')
    parser.add_argument('--buffer-size', type=int, default=10000,
                        help='Buffer size (sentences)')
    parser.add_argument('--max-tokens', type=int, default=12000,
                        help='Maximum number of tokens to process in a batch')
    parser.add_argument('--max-sentences', type=int, default=None,
                        help='Maximum number of sentences to process in a batch')
    parser.add_argument('--cpu-embed', action='store_true',
                        help='Use CPU instead of GPU (for embedding)')
    parser.add_argument('--gpu-kmeans', action='store_true',
                        help='Use GPU instead of CPU (for kmeans)')
    parser.add_argument('--stable', action='store_true',
                        help='Use stable merge sort instead of quick sort')

    parser.add_argument('-n', '--num-clusters', type=int, default=10,
                        help='Number of clusters')
    parser.add_argument('--min-points-per-centroid', type=int, default=39,
                    help='Below, you get a warning')                  
    parser.add_argument('--max-points-per-centroid', type=int, default=256,
                        help='Above, the training set is subsampled')
    parser.add_argument('--nredo', type=int, default=1,
                        help='run the clustering this number of times, '
                        'and keep the best centroids (selected according to clustering objective)')
    parser.add_argument('--niter', type=int, default=25,
                        help='Clustering iterations')
    parser.add_argument('--spherical', action='store_true',
                        help='L2 normalize centroids after each iteration')
    parser.add_argument('--update-index', action='store_true',
                        help='Re-train index after each iteration')                    
    args = parser.parse_args()

    args.buffer_size = max(args.buffer_size, 1)
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    if args.verbose:
        print(' - Encoder: loading {}'.format(args.encoder))
    encoder = SentenceEncoder(args.encoder,
                              max_sentences=args.max_sentences,
                              max_tokens=args.max_tokens,
                              sort_kind='mergesort' if args.stable else 'quicksort',
                              cpu=args.cpu_embed)

    with tempfile.TemporaryDirectory() as tmpdir:
        if args.token_lang != '--':
            tok_fname = os.path.join(tmpdir, 'tok')
            Token(args.input,
                  tok_fname,
                  lang=args.token_lang,
                  romanize=True if args.token_lang == 'el' else False,
                  lower_case=True, gzip=False,
                  verbose=args.verbose, over_write=False)

        if args.bpe_codes:
            bpe_fname = os.path.join(tmpdir, 'bpe')
            BPEfastApply(tok_fname,
                         bpe_fname,
                         args.bpe_codes,
                         verbose=args.verbose, over_write=False)

        ClusterFile(encoder, args.input, bpe_fname, args.output,
                    buffer_size=args.buffer_size,
                    verbose=args.verbose,
                    num_clusters=args.num_clusters,
                    niter=args.niter, nredo=args.nredo,
                    min_points_per_centroid=args.min_points_per_centroid,
                    max_points_per_centroid=args.max_points_per_centroid,
                    spherical=args.spherical,
                    update_index=args.update_index,
                    gpu_kmeans=args.gpu_kmeans)
