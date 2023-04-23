# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

TOLERANCE = {
    "float32": {"atol": 1e-6, "rtol": 1e-6},
    "float16": {"atol": 1e-4, "rtol": 1e-4},
    "bfloat16": {"atol": 1e-3, "rtol": 1e-3},
}


def convert_dtype_to_torch_type(dtype):
    import torch

    if dtype == 'float32':
        return torch.float32
    elif dtype == 'float16':
        return torch.float16
    elif dtype == 'bfloat16':
        return torch.bfloat16


def np_assert_accuracy(
    np_a,
    np_b,
    atol,
    rtol,
    dtype,
    version_a,
    version_b,
    eager_or_static_mode,
    fwd_or_bkd,
    api,
):
    max_atol_idx = np.argmax(np.abs(np_a - np_b))
    max_rtol_idx = np.argmax(np.abs((np_a - np_b) / np_a))
    np.testing.assert_allclose(
        np_a,
        np_b,
        atol,
        rtol,
        err_msg=(
            '{api} {eager_or_static_mode} {fwd_or_bkd}: compare {version_a} res with {version_b} failed in {dtype} dtype,\n'.format(
                api=api,
                eager_or_static_mode=eager_or_static_mode,
                fwd_or_bkd=fwd_or_bkd,
                version_a=version_a,
                version_b=version_b,
                dtype=dtype,
            )
            + 'max_atol_idx: {max_atol_idx}, {version_a}_value: {value_a}, {version_b}_value: {value_b},\n'.format(
                max_atol_idx=max_atol_idx,
                version_a=version_a,
                value_a=str(np_a.flatten()[max_atol_idx].item()),
                version_b=version_b,
                value_b=str(np_b.flatten()[max_atol_idx].item()),
            )
            + 'max_rtol_idx: {max_rtol_idx}, {version_a}_value: {value_a}, {version_b}_value: {value_b},\n'.format(
                max_rtol_idx=max_rtol_idx,
                version_a=version_a,
                value_a=str(np_a.flatten()[max_rtol_idx].item()),
                version_b=version_b,
                value_b=str(np_b.flatten()[max_rtol_idx].item()),
            )
        ),
    )


def np_assert_staility(
    np_actual,
    np_baseline,
    dtype,
    version,
    eager_or_static_mode,
    fwd_or_bkd,
    api,
):
    max_atol_idx = np.argmax(np.abs(np_actual - np_baseline))
    max_rtol_idx = np.argmax(np.abs((np_actual - np_baseline) / np_actual))
    np.testing.assert_equal(
        np_actual,
        np_baseline,
        err_msg=(
            '{eager_or_static_mode} {fwd_or_bkd}: {version} is unstable in {dtype} dtype,\n'.format(
                eager_or_static_mode=eager_or_static_mode,
                fwd_or_bkd=fwd_or_bkd,
                version=version,
                dtype=dtype,
            )
            + 'max_atol_idx: {max_atol_idx}, {version}_value: {actual_value}, {version}_baseline_value: {baseline_value}, \n'.format(
                max_atol_idx=max_atol_idx,
                version=version,
                actual_value=str(np_actual.flatten()[max_atol_idx].item()),
                baseline_value=str(np_baseline.flatten()[max_atol_idx].item()),
            )
            + 'max_rtol_idx: {max_rtol_idx}, {version}_value: {actual_value}, {version}_baseline_value: {baseline_value}, \n'.format(
                max_rtol_idx=max_rtol_idx,
                version=version,
                actual_value=str(np_actual.flatten()[max_rtol_idx].item()),
                baseline_value=str(np_baseline.flatten()[max_rtol_idx].item()),
            )
        ),
    )
