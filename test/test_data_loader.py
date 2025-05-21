# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os, sys
sys.path.insert(0, os.path.abspath('..'))
from torchtitan.datasets import build_hf_data_loader


dp_degree, dp_rank = 1, 0

class TestDataLoader:
    dataset = "willett"
    dataset_path = ""
    batch_size = 2
    seq_len = 131072
    vocab_size = 64

# build dataloader
data_loader = build_hf_data_loader(
    TestDataLoader.dataset,
    TestDataLoader.dataset_path,
    TestDataLoader.batch_size,
    TestDataLoader.seq_len,
    TestDataLoader.vocab_size,
    dp_degree,
    dp_rank,
)

# create iterator
data_iterator = iter(data_loader)

for i in range(33):
    batch = next(data_iterator)
    inputs, labels = batch
    print(f'Input shape: {inputs.shape}')
    print(f'Input dtype: {inputs.dtype}')
    print(f'Input max: {inputs.max()}')
    print(f'Input min: {inputs.min()}')
