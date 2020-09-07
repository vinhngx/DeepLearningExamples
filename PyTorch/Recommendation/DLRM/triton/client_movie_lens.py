#!/usr/bin/env python
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import json
import sys
import pdb

import numpy as np
import torch
import tritonhttpclient
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import pandas as pd
import pickle

from dlrm.data.datasets import SyntheticDataset, SplitCriteoDataset
from dlrm.data.datasets_movie_lens import MovieLensDataset
from torch.utils.data import DataLoader

def get_data_loader(batch_size, *, data_path, model_config):
    with open(model_config.dataset_config) as f:
        categorical_sizes = list(json.load(f).values())
    if data_path:
        data = SplitCriteoDataset(
            data_path=data_path,
            batch_size=batch_size,
            numerical_features=True,
            categorical_features=range(len(categorical_sizes)),
            categorical_feature_sizes=categorical_sizes,
            prefetch_depth=1
        )
    else:
        data = SyntheticDataset(
            num_entries=batch_size * 1024,
            batch_size=batch_size,
            numerical_features=model_config.num_numerical_features,
            categorical_feature_sizes=categorical_sizes,
            device="cpu"
        )

    return torch.utils.data.DataLoader(data,
                                       batch_size=None,
                                       num_workers=0,
                                       pin_memory=False)


def run_infer(model_name, model_version, numerical_features, categorical_features, headers=None):
    inputs = []
    outputs = []
    num_type = "FP16" if numerical_features.dtype == np.float16 else "FP32"
    inputs.append(tritonhttpclient.InferInput('input__0', numerical_features.shape, num_type))
    inputs.append(tritonhttpclient.InferInput('input__1', categorical_features.shape, "INT64"))

    # Initialize the data
    inputs[0].set_data_from_numpy(numerical_features, binary_data=True)
    inputs[1].set_data_from_numpy(categorical_features, binary_data=False)

    outputs.append(tritonhttpclient.InferRequestedOutput('output__0', binary_data=True))
    results = triton_client.infer(model_name,
                                  inputs,
                                  model_version=str(model_version) if model_version != -1 else '',
                                  outputs=outputs,
                                  headers=headers)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--triton-server-url',
                        type=str,
                        required=True,
                        help='URL adress of triton server (with port)')
    parser.add_argument('--triton-model-name', type=str, required=True,
                        help='Triton deployed model name')
    parser.add_argument('--triton-model-version', type=int, default=-1,
                        help='Triton model version')
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-H', dest='http_headers', metavar="HTTP_HEADER",
                        required=False, action='append',
                        help='HTTP headers to add to inference server requests. ' +
                        'Format is -H"Header:Value".')

    parser.add_argument("--dataset_config", type=str, required=True)
    parser.add_argument("--inference_data", type=str,
                        help="Path to file with inference data.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Inference request batch size")
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="Use 16bit for numerical input")

    FLAGS = parser.parse_args()
    try:
        triton_client = tritonhttpclient.InferenceServerClient(url=FLAGS.triton_server_url, verbose=FLAGS.verbose)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    if FLAGS.http_headers is not None:
        headers_dict = {l.split(':')[0]: l.split(':')[1]
                        for l in FLAGS.http_headers}
    else:
        headers_dict = None

    triton_client.load_model(FLAGS.triton_model_name)
    if not triton_client.is_model_ready(FLAGS.triton_model_name):
        sys.exit(1)

    #data_loader_train, data_loader_test = get_data_loaders(FLAGS)
    data_test = MovieLensDataset(data_path='/workspace/dlrm/notebooks/data/ml-25m/test.csv', batch_size=FLAGS.batch_size)

    def collate_fn(batch):
        batch = batch[0]
        return batch

    dataloader = DataLoader(data_test, collate_fn=collate_fn, shuffle=False)

    #dataloader = get_data_loader(FLAGS.batch_size,
    #                             data_path=FLAGS.inference_data,
    #                             model_config=FLAGS)

    MOVIE_LENS_DATA_PATH = './notebooks/data/ml-25m/'
    movie_data = pd.read_csv(MOVIE_LENS_DATA_PATH + "movies_preprocessed_v4.csv", dtype={'imdbId': str})
    test_data = pd.read_csv(MOVIE_LENS_DATA_PATH + "test.csv")
    all_movies_id = movie_data['movieId']

    with open('./notebooks/assets/features.pkl', 'rb') as f:
        cat = pickle.load(f)
        num_feats, cat_feats = cat['numeric_features'], cat['cat_features']

    # prepare test data
    num_samples_test = test_data.shape[0]
    test_data_neg = pd.read_csv(MOVIE_LENS_DATA_PATH + "test.csv").copy()
    test_data_neg['movieId'] = all_movies_id.sample(n=test_data_neg.shape[0], replace=True).values
    test_data = pd.concat([test_data, test_data_neg])
    test_data = test_data.merge(movie_data, on='movieId', how='left')

    test_numerical_features = torch.tensor(test_data[num_feats].values, dtype=torch.float32, device='cuda')
    test_categorical_features = torch.tensor(test_data[cat_feats].values, dtype=torch.int64, device='cuda')
    test_click = torch.cat((torch.ones((num_samples_test), dtype=torch.float32, device='cuda'),
                           torch.zeros((num_samples_test), dtype=torch.float32, device='cuda')), 0)

    results = []
    tgt_list = []

    #steps_per_epoch = int(len(test_data) / FLAGS.batch_size)+1
    #batch_indices = np.array_split(test_data.index, steps_per_epoch)

    n = int(FLAGS.batch_size)
    batch_indices = [test_data.index[i * n:(i + 1) * n] for i in range((len(test_data.index) + n - 1) // n)][:-1]
    steps_per_epoch = len(batch_indices)

    #for numerical_features, categorical_features, target in tqdm(dataloader):
    for step in tqdm(range(steps_per_epoch)):

        #pdb.set_trace()
        numerical_features = test_numerical_features[batch_indices[step],:]
        categorical_features = test_categorical_features[batch_indices[step],:]
        target = test_click[batch_indices[step]]

        numerical_features = numerical_features.cpu().numpy().astype(np.float16 if FLAGS.fp16 else np.float32)
        categorical_features = categorical_features.cpu().numpy()

        output = run_infer(FLAGS.triton_model_name, FLAGS.triton_model_version,
                           numerical_features, categorical_features, headers_dict)

        results.append(output.as_numpy('output__0'))
        tgt_list.append(target.cpu().numpy())

    results = np.concatenate(results).squeeze()
    tgt_list = np.concatenate(tgt_list)

    score = roc_auc_score(tgt_list, results)
    print(F"Model score: {score}")

    statistics = triton_client.get_inference_statistics(model_name=FLAGS.triton_model_name, headers=headers_dict)
    print(statistics)
    if len(statistics['model_stats']) != 1:
        print("FAILED: Inference Statistics")
        sys.exit(1)
