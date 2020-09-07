# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import pickle
import torch


class MovieLensDataset(Dataset):
    """Movie lens dataset."""

    def __init__(
            self,
            data_path='/workspace/dlrm/notebooks/data/ml-25m/train.csv',
            movie_path='/workspace/dlrm/notebooks/data/ml-25m/movies_preprocessed_v4.csv',
            feature_path='/workspace/dlrm/notebooks/assets/features.pkl',
            batch_size=4096
    ):
        self.movie_data = pd.read_csv(movie_path, dtype={'imdbId': str})

        self.data = pd.read_csv(data_path)
        self.num_entries = int(
            2 * self.data.shape[0] / batch_size) + 1  # postive and negetive samples (equal numbers)
        #self.batch_indices = np.array_split(self.data.index, self.num_entries)
        n = int(batch_size / 2)
        self.batch_indices = [self.data.index[i * n:(i + 1) * n] for i in range((len(self.data.index) + n - 1) // n)][
                             :-1]

        self.all_movies_id = self.movie_data['movieId']

        with open(feature_path, 'rb') as f:
            cat = pickle.load(f)
            self.num_feats, self.cat_feats = cat['numeric_features'], cat['cat_features']

        # shuffle train data
        self.data = self.data.reindex(np.random.permutation(self.data.index))

    def __len__(self):
        return self.num_entries

    def __getitem__(self, step):
        if step >= self.num_entries:
            raise IndexError()

        num_samples = len(self.batch_indices[step])

        batch_data = self.data.ix[self.batch_indices[step]]
        batch_data_neg = batch_data.copy()
        batch_data_neg['movieId'] = self.all_movies_id.sample(n=batch_data_neg.shape[0], replace=True).values
        batch_data = pd.concat([batch_data, batch_data_neg])

        batch_data = batch_data.merge(self.movie_data, on='movieId', how='left')

        numerical_features = torch.tensor(batch_data[self.num_feats].values, dtype=torch.float32)
        categorical_features = torch.tensor(batch_data[self.cat_feats].values, dtype=torch.int64)
        click = torch.cat(
            (torch.ones((num_samples), dtype=torch.float32), torch.zeros((num_samples), dtype=torch.float32)), 0)

        return numerical_features, categorical_features, click

    def __del__(self):
        del self.movie_data
        del self.data