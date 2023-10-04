"""A simple partial pivoting (row swap) linear equation system solver based on LU decomposition.

Usually, `tf.linalg.solve`, or `tf.linalg.lu` and `tf.linalg.lu_solve`, are preferable,
but as of TF 2.12, `tf.linalg.solve` and `tf.linalg.lu_solve` do not have a kernel for float16.
Of course float16 doesn't have much accuracy, but in some use cases it can be useful to explore
whether it would be enough, to obtain a possible speedup.

The recommended way is:

  1) Ignore the decomposition routines here and use the much faster `tf.linalg.lu`
     (its outputs `LU` and `p` are compatible with this module).
  2) `tf.cast` your `LU` to the desired dtype.
  3) If the dtype is not supported by the TF kernels, use the `solve` routine from here.

As usual, we separate the solving process into decomposition and triangular solve stages.
This allows reusing the decomposition when solving a new RHS with the same matrix
(or in the batched version, a new set of RHSs with the same set of matrices).

(And as you already know, as a bonus, the LU decomposition gives you the eigenvalues - they are
on the diagonal of the packed LU.)

This is a TensorFlow port of the basic functionality of:
  https://github.com/Technologicat/pylu/blob/master/pylu/dgesv.pxd

This solver is based on high-level TensorFlow operations, so it can run with any supported dtype.
Note this also means it's somewhat slower than the C++ kernels in TF - so this can be used for
testing whether float16 accuracy would be enough for your use case, but the performance is likely
slower than the TF kernel on float32.
"""

__all__ = ["decompose",  # batched, many independent equation systems
           "solve",
           "unpack",
           "decompose_one",  # one equation system
           "solve_one",
           "unpack_one"]

import typing

import tensorflow as tf


# --------------------------------------------------------------------------------
# Batched versions, for many independent equation systems.

def unpack(lu: tf.Tensor, *, validate_args: bool = True) -> typing.Tuple[tf.Tensor, tf.Tensor]:
    """[debug] Decode packed format produced by `decompose`.

    lu: rank-3 tensor, [batch, n, n]. Batch of packed LU decompositions from `decompose`.

    Returns a tuple of rank-2 tensors, `(l, u)`.
    """
    shape = tf.shape(lu)
    if validate_args and (len(shape) != 3 or int(shape[1]) != int(shape[2])):
        raise ValueError(f"Expected `lu` to be a tensor of shape [batch, n, n], got {shape}")

    l = tf.Variable(lu, name="l", trainable=False)  # noqa: E741, TF prefers lowercase identifiers.
    u = tf.Variable(lu, name="u", trainable=False)
    unpack_kernel(l, u)
    return l, u

@tf.function
def unpack_kernel(l: tf.Tensor, u: tf.Tensor):  # noqa: E741, TF prefers lowercase identifiers.
    n = int(tf.shape(l)[-1])

    # This computation batches trivially.
    for i in range(n):
        l[:, i, i].assign(tf.cast(1.0, l.dtype))
        for j in range(i + 1, n):
            l[:, i, j].assign(tf.cast(0.0, l.dtype))

        for j in range(i):
            u[:, i, j].assign(tf.cast(0.0, u.dtype))


def decompose(a: tf.Tensor, validate_args: bool = True) -> typing.Tuple[tf.Tensor, tf.Tensor]:
    """LU decompose a batch of `n × n` matrices, using partial pivoting (row swaps).

    Produces matrices `L` and `U`, and a row-swapping permutation matrix `P`, such that::

      P A = L U

    The permutation matrix `P` is actually encoded as a permutation vector `p`.

    `A`: rank-3 tensor, [batch, n, n], C storage order. A batch of square matrices to be LU decomposed.

    Returns: `(lu, p)`, where
       `lu`: rank-3 tensor, [batch, n, n]; the `L` and `U` matrices.

             The storage format is packed so that the diagonal elements of `L` are implicitly 1 (not stored).
             The diagonal of `lu` stores the diagonal elements of `U`.

             That is, `lu = L + U - eye`.

       `p`: rank-2 tensor, [batch, n]; permutation vectors used in partial pivoting.
    """
    shape = tf.shape(a)
    if validate_args and (len(shape) != 3 or int(shape[1]) != int(shape[2])):
        raise ValueError(f"Expected `a` to be a tensor of shape [batch, n, n], got {shape}")
    batch = int(shape[0])
    n = int(shape[1])

    # TODO: fix data placement woes with `jit_compile=True`; this doesn't help: with tf.device('GPU:0'):

    lu = tf.Variable(a, name="lu", trainable=False)  # copy `a` into a variable `lu`, which we will decompose in-place
    p = tf.range(n)
    p = tf.expand_dims(p, axis=0)  # [1, n]
    p = tf.tile(p, [batch, 1])  # [batch, n]
    p = tf.Variable(p, name="p", trainable=False)

    wrkf = tf.Variable(tf.zeros([batch], dtype=a.dtype), name="wrkf")
    wrki = tf.Variable(tf.zeros([batch], dtype=tf.int32), name="wrki")

    decompose_kernel(lu, p, wrkf, wrki)

    return lu, p


# wrkf and wrki are float and integer work space vectors, respectively.
@tf.function
def decompose_kernel(lu: tf.Variable, p: tf.Variable,
                     wrkf: tf.Variable, wrki: tf.Variable) -> None:
    """The actual compute kernel. Modifies `lu` and `p` in-place."""
    shape = tf.shape(lu)
    batch = int(shape[0])
    n = int(shape[1])

    for k in range(n):
        # Pivot: find row `r` such that `|a_rk| = max( |a_sk|, s = k, k+1, ..., n-1 )` (note 0-based indexing)
        wrkf[:].assign(-1.0)  # max magnitude, for each `lu` in batch; any invalid value
        wrki[:].assign(-1)  # row `r`, for each `lu` in batch
        for s in range(k, n):
            mag = tf.abs(lu[:, s, k])  # [batch]
            wrki.assign(tf.where(tf.greater(mag, wrkf), s, wrki))  # if mag > max_mag then s, else old r
            wrkf.assign(tf.where(tf.greater(mag, wrkf), mag, wrkf))  # update max_mag

        # Swap elements `k` and `r` of permutation vector `p`
        tmp1 = p[:, k]
        for m in range(batch):  # TODO: tensorize over batch
            p[m, k].assign(p[m, wrki[m]])  # p[m, k] = p[m, r]
            p[m, wrki[m]].assign(tmp1[m])  # p[m, r] = tmp1[m]  (old p[m, k])

        # Physically swap also the corresponding rows of A.
        #
        # This may seem a silly way to do it (why not indirect via p?), but it makes future accesses faster,
        # because then we won't need to use the permutation vector p when accessing A or lu.
        #
        # (It is still needed for accessing the RHS vector when solving the actual equation system.)
        #
        tmp2 = lu[:, k, :]
        for m in range(batch):  # TODO: tensorize over batch
            lu[m, k, :].assign(lu[m, wrki[m], :])  # lu[m, k, :] = lu[m, r, :]
            lu[m, wrki[m], :].assign(tmp2[m, :])  # lu[m, r, :] = tmp2[m, :]  (old lu[m, k, :])

        # # TODO: Fail gracefully if the pivoted lu[k, k] is below some tolerance
        # # TODO: make the tolerance depend on dtype
        # if lu[k, k] <= 1e-6:
        #     return False

        # This actual computation here batches trivially.
        for i in range(k + 1, n):
            lu[:, i, k].assign(lu[:, i, k] / lu[:, k, k])
            for j in range(k + 1, n):
                lu[:, i, j].assign(lu[:, i, j] - lu[:, i, k] * lu[:, k, j])


def solve(lu: tf.Tensor, p: tf.Tensor, b: tf.Tensor, *, validate_args: bool = False) -> tf.Tensor:
    """Solve a batch of linear equation systems after the matrices have been LU-decomposed.

    `lu`: rank-3 tensor, [batch, n, n], C storage order. Batch of packed LU decompositions from `decompose`.
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
    shape = tf.shape(lu)
    if validate_args and (len(shape) != 3 or int(shape[1]) != int(shape[2])):
        raise ValueError(f"Expected `a` to be a tensor of shape [batch, n, n], got {shape}")
    batch = int(shape[0])
    n = int(shape[1])

    if validate_args:
        for name, tensor in (("p", p), ("b", b)):
            shape = tf.shape(tensor)
            if len(shape) != 2 or int(shape[1]) != n:
                raise ValueError(f"Expected {name} to be a tensor of shape [batch, n], and from `lu`, n = {n}; got {shape}")

    x = tf.Variable(tf.zeros([batch, n], dtype=lu.dtype), name="x")
    solve_kernel(lu, p, b, x)
    return x


@tf.function
def solve_kernel(lu: tf.Tensor, p: tf.Tensor, b: tf.Tensor, x: tf.Variable) -> None:
    """The actual compute kernel. Writes `x` in-place."""
    shape = tf.shape(lu)
    # batch = int(shape[0])
    n = int(shape[1])

    # TODO: The permutation operator should be created in `decompose` instead.
    P = tf.linalg.LinearOperatorPermutation(p, dtype=lu.dtype, name="P")
    bmat = tf.expand_dims(b, axis=-1)
    x.assign(tf.squeeze(P.matmul(bmat), axis=-1))  # x[:] = b[p], or formally, x := P b

    # Solve `L y = P b` by forward substitution.
    for i in range(n):
        for j in range(i):
            x[:, i].assign(x[:, i] - lu[:, i, j] * x[:, j])
    # Now `x` contains the solution of the first equation, i.e. it is actually `y`.

    # Solve `U x = y` by backward substitution.
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            x[:, i].assign(x[:, i] - lu[:, i, j] * x[:, j])
        x[:, i].assign(x[:, i] / lu[:, i, i])
    # Now the array "x" contains the solution of the second equation, i.e. x.


# --------------------------------------------------------------------------------
# Non-batched versions, for one equation system.
#
# These are here mainly because the code is much more readable than the batched version,
# and more readily comparable to `dgesv.pxd` in PyLU.

def unpack_one(lu: tf.Tensor, validate_args: bool = True) -> typing.Tuple[tf.Tensor, tf.Tensor]:
    """[debug] Decode packed format produced by `decompose_one`.

    lu: rank-2 tensor, [n, n]. Packed LU decomposition from `decompose_one`.

    Returns a tuple of rank-2 tensors, `(l, u)`.
    """
    shape = tf.shape(lu)
    if validate_args and (len(shape) != 2 or int(shape[0]) != int(shape[1])):
        raise ValueError(f"Expected `lu` to be a tensor of shape [n, n], got {shape}")

    l = tf.Variable(lu, name="l", trainable=False)  # noqa: E741, TF prefers lowercase identifiers.
    u = tf.Variable(lu, name="u", trainable=False)
    unpack_one_kernel(l, u)
    return l, u

@tf.function
def unpack_one_kernel(l: tf.Tensor, u: tf.Tensor):  # noqa: E741, TF prefers lowercase identifiers.
    n = int(tf.shape(l)[-1])

    for i in range(n):
        l[i, i].assign(tf.cast(1.0, l.dtype))
        for j in range(i + 1, n):
            l[i, j].assign(tf.cast(0.0, l.dtype))

        for j in range(i):
            u[i, j].assign(tf.cast(0.0, u.dtype))


def decompose_one(a: tf.Tensor, validate_args: bool = True) -> typing.Tuple[tf.Tensor, tf.Tensor]:
    """LU decompose an `n × n` matrix, using partial pivoting (row swaps).

    Produces matrices `L` and `U`, and a row-swapping permutation matrix `P`, such that::

      P A = L U

    The permutation matrix `P` is actually encoded as a permutation vector `p`.

    `A`: rank-2 tensor, [n, n], C storage order. A square matrix to be LU decomposed.

    Returns: `(lu, p)`, where
       `lu`: rank-2 tensor, [n, n]; the `L` and `U` matrices.

             The storage format is packed so that the diagonal elements of `L` are implicitly 1 (not stored).
             The diagonal of `lu` stores the diagonal elements of `U`.

             That is, `lu = L + U - eye`.

       `p`: rank-1 tensor, [n]; permutation vector used in partial pivoting.
    """
    shape = tf.shape(a)
    if validate_args and (len(shape) != 2 or int(shape[0]) != int(shape[1])):
        raise ValueError(f"Expected `a` to be a tensor of shape [n, n], got {shape}")
    n = int(shape[0])

    # TODO: data placement woes; this doesn't help: with tf.device('GPU:0'):

    lu = tf.Variable(a, name="lu", trainable=False)  # copy `a` into a variable `lu`, which we will decompose in-place
    p = tf.Variable(tf.range(n), name="p", trainable=False)
    decompose_one_kernel(lu, p)

    return lu, p


@tf.function
def decompose_one_kernel(lu: tf.Variable, p: tf.Variable) -> None:
    """The actual compute kernel. Modifies `lu` and `p` in-place."""
    n = int(tf.shape(lu)[-1])

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
        # (It is still needed for accessing the RHS vector when solving the actual equation system.)
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


def solve_one(lu: tf.Tensor, p: tf.Tensor, b: tf.Tensor, *, validate_args: bool = False) -> tf.Tensor:
    """Solve a linear equation system after the matrix has been LU-decomposed.

    `lu`: rank-2 tensor, [n, n], C storage order. Packed LU decomposition from `decompose_one`.
    `p`: rank-1 tensor, [n]. Permutation vector from `decompose`.
    `b`: rank-1 tensor, [n]. Right-hand side of `A x = b`.

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

    Returns `x`, a rank-1 tensor [n]; solution for `P A x = P b`.
    """
    shape = tf.shape(lu)
    if validate_args and (len(shape) != 2 or int(shape[0]) != int(shape[1])):
        raise ValueError(f"Expected `a` to be a tensor of shape [n, n], got {shape}")
    n = int(shape[0])

    if validate_args:
        for name, tensor in (("p", p), ("b", b)):
            shape = tf.shape(tensor)
            if len(shape) != 1 or int(shape[0]) != n:
                raise ValueError(f"Expected {name} to be a tensor of shape [n], and from `lu`, n = {n}; got {shape}")

    x = tf.Variable(tf.zeros([n], dtype=lu.dtype), name="x")
    solve_one_kernel(lu, p, b, x)
    return x


@tf.function
def solve_one_kernel(lu: tf.Tensor, p: tf.Tensor, b: tf.Tensor, x: tf.Variable) -> None:
    """The actual compute kernel. Writes `x` in-place."""
    n = int(tf.shape(lu)[-1])

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
