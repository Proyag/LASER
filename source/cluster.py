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
def WriteBatch(sentences, raw_sentences,
               cluster_dists, cluster_ids,
               out_file, output_ids=False, output_distances=False):
    assert len(raw_sentences) == len(sentences) == cluster_ids.shape[0] == cluster_dists.shape[0], \
        'Incorrect number of cluster ids/sentences'
    if output_ids:
        for cluster_id in cluster_ids:
            out_file.write(str(cluster_id[0]) + '\n')
    elif output_distances:
        # Need distances corresponding to clusters, instead of sorted ascending like faiss provides
        for idx, cluster_dist in enumerate(cluster_dists):
            unsorted_dist = np.empty(cluster_dist.shape)
            for pos, dist in enumerate(cluster_dist):
                unsorted_dist[cluster_ids[idx][pos]] = cluster_dist[pos]
            out_file.write(str(' '.join(str(x) for x in unsorted_dist.tolist())) + '\n')
    else:
        for sentence, raw_sentence, cluster_id in zip(sentences, raw_sentences, cluster_ids):
            out_file[cluster_id].write(raw_sentence + '\n')    


# Cluster sentences (existing file pointers)
def ClusterFilep(encoder, raw_inp_file, bpe_inp_file, out_files, centroids_fname,
                 buffer_size=10000, verbose=False,
                 output_ids=False, output_distances=False,
                 num_clusters=10, niter=25, nredo=1,
                 no_reload_centroids=False,
                 min_points_per_centroid=39, max_points_per_centroid=256,
                 spherical=False, update_index=False, gpu_kmeans=False):
    n = 0
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
                if not no_reload_centroids and os.path.exists(centroids_fname):
                    print(' - Clustering: Loading centroids from {}'.format(centroids_fname))
                    centroids_loaded = np.load(centroids_fname)
                    index = faiss.IndexFlatL2(LASER_DIM)
                    index.add(centroids_loaded)
                else:
                    kmeans.train(encoded)
                    np.save(centroids_fname, kmeans.centroids, allow_pickle=False)
                    index = kmeans.index
                cluster_dists, cluster_ids = index.search(encoded, num_clusters)
                WriteBatch(sentences, raw_sentences,
                           cluster_dists, cluster_ids,
                           out_files, output_ids, output_distances)
        else:
            cluster_dists, cluster_ids = index.search(encoder.encode_sentences(sentences), num_clusters)
            WriteBatch(sentences, raw_sentences,
                       cluster_dists, cluster_ids,
                       out_files, output_ids, output_distances)
        if verbose and n % 10000 == 0:
            print('\r - Clustering: {:d} sentences'.format(n), end='')
    if verbose:
        print('\r - Clustering: Clustered {:d} sentences'.format(n), end='')
        EncodeTime(t)            


# Cluster sentences (file names)
def ClusterFile(encoder, raw_fname, bpe_fname, out_fname, inp_encoding='utf-8',
                buffer_size=10000, verbose=False, output_ids=False, output_distances=False,
                num_clusters=10, niter=25, nredo=1, no_reload_centroids=False,
                min_points_per_centroid=39, max_points_per_centroid=256,
                spherical=False, update_index=False, gpu_kmeans=False):
    if verbose:
        print(' - Encoder: Clustering {} to {}'.
              format(os.path.basename(bpe_fname) if len(bpe_fname) > 0 else 'stdin',
                     os.path.basename(out_fname)))
    fin_raw = open(raw_fname, 'r', encoding=inp_encoding, errors='surrogateescape')
    fin_bpe = open(bpe_fname, 'r', encoding=inp_encoding, errors='surrogateescape')
    if output_ids or output_distances:
        fout = open(out_fname, mode='w')
    else:
        fout = [open(out_fname + '.cluster_{}'.format(i), mode='w') for i in range(num_clusters)]
    centroids_fname = out_fname + '.centroids.npy'
    ClusterFilep(encoder, fin_raw, fin_bpe, fout, centroids_fname,
                 buffer_size=buffer_size, verbose=verbose,
                 output_ids=output_ids, output_distances=output_distances,
                 num_clusters=num_clusters, niter=niter, nredo=nredo,
                 min_points_per_centroid=min_points_per_centroid,
                 max_points_per_centroid=max_points_per_centroid,
                 spherical=spherical, update_index=update_index, gpu_kmeans=gpu_kmeans)
    fin_raw.close()
    fin_bpe.close()
    if output_ids or output_distances:
        fout.close()
    else:
        for f in fout:
            f.close()


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
                        help='Output file prefix / '
                             'File to write cluster IDs with --output-cluster-ids-only')
    parser.add_argument('--output-cluster-ids-only', action='store_true',
                        help='Output only cluster IDs instead of writing sentences to separate files')
    parser.add_argument('--output-cluster-distances-only', action='store_true',
                        help='Output only distances from each cluster centroid')
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
    parser.add_argument('--no-reload-centroids', action='store_true',
                        help='Do not reload existing centroids and force Kmeans retraining')
    args = parser.parse_args()

    args.buffer_size = max(args.buffer_size, 1)
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    assert not (args.output_cluster_ids_only and args.output_cluster_distances_only), \
        '--output-cluster-ids-only and --output-cluster-distances-only cannot both be active'

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
                    output_ids=args.output_cluster_ids_only,
                    output_distances=args.output_cluster_distances_only,
                    num_clusters=args.num_clusters,
                    niter=args.niter,
                    nredo=args.nredo,
                    no_reload_centroids=args.no_reload_centroids,
                    min_points_per_centroid=args.min_points_per_centroid,
                    max_points_per_centroid=args.max_points_per_centroid,
                    spherical=args.spherical,
                    update_index=args.update_index,
                    gpu_kmeans=args.gpu_kmeans)
