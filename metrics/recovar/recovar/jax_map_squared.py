# Copyright 2019 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


## Modified from here https://jax.readthedocs.io/en/latest/_modules/jax/_src/scipy/ndimage.html#map_coordinates to just square the weights of the interpolator. This is useful for computing the diagonal of the A^TA matrix fast.
## Probably should rewrite this


from collections.abc import Sequence
import functools
import itertools
import operator
import textwrap
from typing import Callable

import scipy.ndimage

from jax._src import api
from jax._src import util
from jax import lax
import jax.numpy as jnp
from jax._src.numpy.util import _wraps
from jax._src.typing import ArrayLike, Array
from jax._src.util import safe_zip as zip


def _nonempty_prod(arrs: Sequence[Array]) -> Array:
  return functools.reduce(operator.mul, arrs)

def _nonempty_sum(arrs: Sequence[Array]) -> Array:
  return functools.reduce(operator.add, arrs)

def _mirror_index_fixer(index: Array, size: int) -> Array:
    s = size - 1 # Half-wavelength of triangular wave
    # Scaled, integer-valued version of the triangular wave |x - round(x)|
    return jnp.abs((index + s) % (2 * s) - s)

def _reflect_index_fixer(index: Array, size: int) -> Array:
    return jnp.floor_divide(_mirror_index_fixer(2*index+1, 2*size+1) - 1, 2)

_INDEX_FIXERS: dict[str, Callable[[Array, int], Array]] = {
    'constant': lambda index, size: index,
    'nearest': lambda index, size: jnp.clip(index, 0, size - 1),
    'wrap': lambda index, size: index % size,
    'mirror': _mirror_index_fixer,
    'reflect': _reflect_index_fixer,
}


def _round_half_away_from_zero(a: Array) -> Array:
  return a if jnp.issubdtype(a.dtype, jnp.integer) else lax.round(a)


def _nearest_indices_and_weights(coordinate: Array) -> list[tuple[Array, ArrayLike]]:
  index = _round_half_away_from_zero(coordinate).astype(jnp.int32)
  weight = coordinate.dtype.type(1)
  return [(index, weight)]


def _linear_indices_and_weights(coordinate: Array) -> list[tuple[Array, ArrayLike]]:
  lower = jnp.floor(coordinate)
  upper_weight = coordinate - lower
  lower_weight = 1 - upper_weight
  index = lower.astype(jnp.int32)
  return [(index, lower_weight**2), (index + 1, upper_weight**2)]


@functools.partial(api.jit, static_argnums=(2, 3, 4))
def _map_coordinates(input: ArrayLike, coordinates: Sequence[ArrayLike],
                     order: int, mode: str, cval: ArrayLike) -> Array:
  input_arr = jnp.asarray(input)
  coordinate_arrs = [jnp.asarray(c) for c in coordinates]
  cval = jnp.asarray(cval, input_arr.dtype)

  if len(coordinates) != input_arr.ndim:
    raise ValueError('coordinates must be a sequence of length input.ndim, but '
                     '{} != {}'.format(len(coordinates), input_arr.ndim))

  index_fixer = _INDEX_FIXERS.get(mode)
  if index_fixer is None:
    raise NotImplementedError(
        'jax.scipy.ndimage.map_coordinates does not yet support mode {}. '
        'Currently supported modes are {}.'.format(mode, set(_INDEX_FIXERS)))

  if mode == 'constant':
    is_valid = lambda index, size: (0 <= index) & (index < size)
  else:
    is_valid = lambda index, size: True

  if order == 0:
    interp_fun = _nearest_indices_and_weights
  elif order == 1:
    interp_fun = _linear_indices_and_weights
  else:
    raise NotImplementedError(
        'jax.scipy.ndimage.map_coordinates currently requires order<=1')

  valid_1d_interpolations = []
  for coordinate, size in zip(coordinate_arrs, input_arr.shape):
    interp_nodes = interp_fun(coordinate)
    valid_interp = []
    for index, weight in interp_nodes:
      fixed_index = index_fixer(index, size)
      valid = is_valid(index, size)
      valid_interp.append((fixed_index, valid, weight))
    valid_1d_interpolations.append(valid_interp)

  outputs = []
  for items in itertools.product(*valid_1d_interpolations):
    indices, validities, weights = util.unzip3(items)
    if all(valid is True for valid in validities):
      # fast path
      contribution = input_arr[indices]
    else:
      all_valid = functools.reduce(operator.and_, validities)
      contribution = jnp.where(all_valid, input_arr[indices], cval)
    outputs.append(_nonempty_prod(weights) * contribution)
  result = _nonempty_sum(outputs)
  if jnp.issubdtype(input_arr.dtype, jnp.integer):
    result = _round_half_away_from_zero(result)
  return result.astype(input_arr.dtype)


@_wraps(scipy.ndimage.map_coordinates, lax_description=textwrap.dedent("""\
    Only nearest neighbor (``order=0``), linear interpolation (``order=1``) and
    modes ``'constant'``, ``'nearest'``, ``'wrap'`` ``'mirror'`` and ``'reflect'`` are currently supported.
    Note that interpolation near boundaries differs from the scipy function,
    because we fixed an outstanding bug (https://github.com/scipy/scipy/issues/2640);
    this function interprets the ``mode`` argument as documented by SciPy, but
    not as implemented by SciPy.
    """))
def map_coordinates_squared(
    input: ArrayLike, coordinates: Sequence[ArrayLike], order: int, mode: str = 'constant', cval: ArrayLike = 0.0,
):
  return _map_coordinates(input, coordinates, order, mode, cval)