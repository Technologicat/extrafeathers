"""A simple partial pivoting (row swap) solver.

This is a TensorFlow port of the basic functionality in:
  https://github.com/Technologicat/pylu/blob/master/pylu/dgesv.pxd

The point is to allow any dtype in some simple use cases; as of TF 2.12,
`tf.linalg.solve` does not have a kernel for float16.

This solver, instead, is based on high-level TensorFlow operations,
so that it can be run with any supported dtype.
"""

__all__ = ["decompose",
           "solve"]

import typing

import tensorflow as tf


def decompose(a: tf.Tensor) -> typing.Tuple[tf.Tensor, tf.Tensor]:
    """LU decompose an `n × n` matrix, using partial pivoting (row swaps).

    Produces matrices `L` and `U` such that `P A = L U`, where `P` is a row-swapping permutation matrix.

    `A`: rank-3 tensor, [batch, n, n], C storage order. A batch of square matrices to be LU decomposed.

    Returns: `(LU, p)`, where
       `LU`: rank-3 tensor, [batch, n, n]; the `L` and `U` matrices.

             The storage format is packed so that the diagonal elements of `L` are implicitly 1 (not stored).
             The diagonal of `LU` stores the diagonal elements of `U`.

       `p`: rank-2 tensor, [batch, n]; permutation vectors used in partial pivoting.
    """
    n = int(tf.shape(a)[-1])

    # TODO: batch support

    # TODO: data placement woes; this doesn't help: with tf.device('GPU:0'):

    lu = tf.Variable(a, name="lu", trainable=False)  # copy `a` into a variable `LU`, which we will decompose in-place
    p = tf.Variable(tf.range(n), name="p", trainable=False)
    decompose_kernel(lu, p)

    return lu, p


@tf.function
def decompose_kernel(lu: tf.Variable, p: tf.Variable) -> None:
    """The actual compute kernel. Modifies `lu` in-place."""
    n = int(tf.shape(lu)[-1])

    # TODO: batch support

    for k in range(n):
        # Pivot: find row `r` such that `|a_rk| = max( |a_sk|, s = k, k+1, ..., n-1 )` (note 0-based indexing)
        max_mag = tf.cast(-1.0, lu.dtype)  # any invalid value
        r = -1
        for s in range(k, n):
            mag = tf.abs(lu[s, k])
            if mag > max_mag:
                max_mag = mag
                r = s

        # Swap elements `k` and `r` of permutation vector `p`
        tmp1 = p[k]
        p[k].assign(p[r])
        p[r].assign(tmp1)

        # Physically swap also the corresponding rows of A.
        #
        # This may seem a silly way to do it (why not indirect via p?), but it makes future accesses faster,
        # because then we won't need to use the permutation vector p when accessing A or lu.
        #
        # (It is still needed for accessing the load vector when solving the actual equation system.)
        #
        tmp2 = lu[k, :]
        lu[k, :].assign(lu[r, :])
        lu[r, :].assign(tmp2)

        # # TODO: Fail gracefully if the pivoted lu[k, k] is below some tolerance
        # # TODO: make the tolerance depend on dtype
        # if lu[k, k] <= 1e-6:
        #     return False

        for i in range(k + 1, n):
            lu[i, k].assign(lu[i, k] / lu[k, k])
            for j in range(k + 1, n):
                lu[i, j].assign(lu[i, j] - lu[i, k] * lu[k, j])


def solve(lu: tf.Tensor, p: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    """Solve a linear equation system after the matrix has been LU-decomposed.

    `lu`: rank-3 tensor, [batch, n, n], C storage order. Packed LU decomposition from `decompose`.
    `p`: rank-2 tensor, [batch, n]. Permutation vector from `decompose`.
    `b`: rank-2 tensor, [batch, n]. Right-hand side of `A x = b`.

    The equation to be solved is::

      A x = b

    in the form::

      P A x = P b

    where P is a permutation matrix (that permutes rows), and `P A = L U`. Hence::

      L U x = P b

    This equation is equivalent with the system::

      L y = P b
      U x = y

    We first solve `L y = P b` and then `U x = y`.

    Returns `x`, a rank-2 tensor [batch, n]; solution for `P A x = P b`.
    """
    n = int(tf.shape(lu)[-1])

    # TODO: batch support

    x = tf.Variable(tf.zeros([n], name="x", dtype=lu.dtype))
    solve_kernel(lu, p, b, x)
    return x


@tf.function
def solve_kernel(lu: tf.Tensor, p: tf.Tensor, b: tf.Tensor, x: tf.Variable) -> None:
    """The actual compute kernel. Writes `x` in-place."""
    n = int(tf.shape(lu)[-1])

    # TODO: batch support

    # Solve `L y = P b` by forward substitution.
    for i in range(n):
        x[i].assign(b[p[i]])  # formally, `(P b)[i]`
        for j in range(i):
            x[i].assign(x[i] - lu[i, j] * x[j])
    # Now `x` contains the solution of the first equation, i.e. it is actually `y`.

    # Solve `U x = y` by backward substitution.
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            x[i].assign(x[i] - lu[i, j] * x[j])
        x[i].assign(x[i] / lu[i, i])
    # Now the array "x" contains the solution of the second equation, i.e. x.
