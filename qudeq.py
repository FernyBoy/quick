# Copyright [2024] Luis Alberto Pineda Cortés & Rafael Morales Gamboa.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import numpy as np
import constants


class QuDeq:
    """Quantizer/dequantizer class.

    Provides the methods for quantizing and dequantizing data on the basis
    of an original corpus and a given number of discrete values. Quantization
    and dequantization is done using minima and maxima data per column.
    """

    def __init__(self, corpus: np.ndarray, percentiles=False):
        self.minima, self.maxima = self.get_min_max(corpus, percentiles)
        idx = np.where(self.minima == self.maxima)[0]
        if len(idx) > 0:
            print(
                f'Minima and maxima have the same value in position(s): {idx.tolist()}'
            )

    def get_min_max(self, a: np.ndarray, percentiles: bool):
        """Produces desirable minimum and maximum values for features."""
        if percentiles:
            return np.percentile(
                a, constants.minimum_percentile, axis=0
            ), np.percentile(a, constants.maximum_percentile, axis=0)
        else:
            return np.min(a, axis=0), np.max(a, axis=0)

    def quantize(self, a: np.ndarray, m: int):
        if a.ndim > 2:
            raise ValueError(f'The array as more than two dimensions: {a.shape}.')
        elif a.ndim == 1:
            b = [
                self._quantize(x, min, max, m)
                for x, min, max in zip(a, self.minima, self.maxima)
            ]
            return np.array(b, dtype=int)
        else:
            b = [self.quantize(e, m) for e in a]
            return np.array(b)

    def dequantize(self, a: np.array, m: int):
        if a.ndim > 2:
            raise ValueError(f'The array as more than two dimensions: {a.shape}.')
        elif a.ndim == 1:
            b = [
                self._dequantize(x, min, max, m)
                for x, min, max in zip(a, self.minima, self.maxima)
            ]
            return np.array(b, dtype=float)
        else:
            b = [self.dequantize(e, m) for e in a]
            return np.array(b)

    def _quantize(self, x, min, max, m):
        if max == min:
            return round((m - 1) / 2)
        elif math.isnan(x):
            return max + 1
        else:
            return round((m - 1) * (x - min) / (max - min))

    def _dequantize(self, i, min, max, m):
        return (max - min) / 2 if m == 1 else (max - min) * i / (m - 1) + min
