# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

## copy from: https://github.com/facebookresearch/astmt/blob/4d5d9bf96b94040ce5be4b12fdccd871e58b1cd3/fblib/util/model_resources/num_parameters.py#L9


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)