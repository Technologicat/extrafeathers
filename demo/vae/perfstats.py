"""Performance statistics for the CVAE model.

This filename was chosen for brevity and clarity - "perf" obviously stands
for "perforated", while `stat (2)` is the *nix system call.
"""

__all__ = ["active_units",
           "negative_log_likelihood",
           "mutual_information"]

import tensorflow as tf

from .config import latent_dim
from .cvae import log_normal_pdf, log_px_z
from .util import batched

# --------------------------------------------------------------------------------
# Helper functions

# TODO: refactor `logsumxs` to `util`
def logsumxs(logxs):
    """`log(∑k x[k])`, computed in terms of `log x[k]`, without evaluating `x[k]`.

    `logxs`: rank-1 tensor, containing `[log x[0], log x[1], ...]`

    Returns `log(∑k x[k])`.

    This is mostly numerically stable, and particularly useful when only the
    logarithms are available (e.g. to prevent overflow from very large exponents).

    **Background**:

    Discussion summarized from:
      https://cdsmithus.medium.com/the-logarithm-of-a-sum-69dd76199790

    For all `x, y > 0`, it holds that:

      log(x + y) = log(x * (1 + y / x))
                 = log x + log(1 + y / x)
                 = log x + log(1 + exp(log(y / x)))
                 = log x + log(1 + exp(log y - log x))
                 ≡ log x + ⟦log y - log x⟧+
    where

      ⟦x⟧+ ≡ log(1 + exp(x))

    is the `softplus` function (notation in analogy with positive part "[x]+").
    Actually, let us define the following C∞ continuous analog of the `max` function:

      smoothmax(x, y) := x + ⟦y - x⟧+

    This gives us the *smoothmax identity* for the logarithm of a sum:

      log(x + y) = smoothmax(log x, log y)

    Upon close inspection, most of the usual properties of addition
    (commutativity, associativity, distributivity) hold for `smoothmax`.

    Commutativity:

      smoothmax(log x, log y) = log(x + y)   (by the above)
                              = log(y + x)   (commutativity of addition)
                              = smoothmax(log y, log x)

    Since `log x` and `log y` are arbitrary real numbers, we may as well rewrite
    the first and last forms as:

      smoothmax(x, y) = smoothmax(y, x)

    which was to be shown.

    Associativity:

        smoothmax(log x, smoothmax(log y, log z))
      = log(x + exp(smoothmax(log y, log z)))
      = log(x + exp(log(y + z)))
      = log(x + y + z)
      = log(exp(log(x + y)) + z)
      = log(exp(smoothmax(log x, log y)) + z)
      = smoothmax(smoothmax(log x, log y), log z)

    so

      smoothmax(x, smoothmax(y, z)) = smoothmax(smoothmax(x, y), z)

    as claimed.

    Distributivity (over addition):

      log x + smoothmax(log y, log z) = log x + log(y + z)
                                      = log(x (y + z))
                                      = log(x y + x z)
                                      = smoothmax(log(x y), log(x z))
                                      = smoothmax(log x + log y, log x + log z)

    Since `log x`, `log y` and `log z` are arbitrary real numbers, we have:

      x + smoothmax(y, z) = smoothmax(x + y, x + z)

    and we see `smoothmax` is distributive (over addition; like addition is
    distributive over multiplication).

    The one to watch out for is the identity property. Since we assumed
    `x, y > 0` to keep all arguments in the domain of (real-valued)
    `log`, strictly speaking the identity is not applicable when `y = 0`.
    In the limit, though, we have:

      lim[y → -∞] smoothmax(x, y) = x

    so we *can* say that:

      lim[y → 0+] log(x + y) = lim[y → 0+] smoothmax(log x, log y)
                             = log x

    Also keep in mind `smoothmax` is not `max`, although it behaves
    somewhat similarly when the arguments are far apart. It differs
    from `max` the most when `|x - y|` is small. The extreme case is:

      smoothmax(x, x) = x + ⟦0⟧+
                      = x + log(1 + exp(0))
                      = x + log(1 + 1)
                      = x + log(2)

    Finally, observe that:

      x + log 1 = x
      x + log 2 = smoothmax(x, x)
      x + log 3 = smoothmax(x, smoothmax(x, x))
      x + log 4 = smoothmax(x, smoothmax(x, smoothmax(x, x)))
      ...

    Proof by induction, omitted. The original author writes:
      [This] resembles a sort of definition of addition of log-naturals
      as “repeated smoothmax of a number with itself”, in very much the
      same sense that multiplication by naturals can be defined as
      repeated addition of a number with itself, strengthening the
      notion that this operation is sort-of one order lower than addition.

    This perhaps looks clearer if we use some symbol, say `𝕄`,
    as infix notation for `smoothmax`:

      x + log 1 = x
      x + log 2 = x 𝕄 x
      x + log 3 = x 𝕄 (x 𝕄 x)
      x + log 4 = x 𝕄 (x 𝕄 (x 𝕄 x))
      ...

    and since `smoothmax` is associative, we can drop the parentheses:

      x + log 1 = x
      x + log 2 = x 𝕄 x
      x + log 3 = x 𝕄 x 𝕄 x
      x + log 4 = x 𝕄 x 𝕄 x 𝕄 x
      ...

    which indeed looks similar to

      x * 1 = x
      x * 2 = x + x
      x * 3 = x + x + x
      x * 4 = x + x + x + x
      ...

    In analogy with the `log` of a product:

      log(x y) = log x + log y

    the `𝕄` notation also gives a pretty expression for the `log` of a sum:

      log(x + y) = log x 𝕄 log y

    where

      x 𝕄 y := x + ⟦y - x⟧+    (smoothmax)
      ⟦x⟧+ := log(1 + exp(x))   (softplus)
    """
    # The benefit of the smoothmax identity is that it allows us to work with
    # logarithms only, except in the evaluation of the softplus. Whenever its
    # argument is small, the `exp` can be taken without numerical issues.
    #
    # Although there is a risk of catastrophic cancellation in `log y - log x`,
    # we still want a reasonable amount of cancellation, to keep the argument
    # to softplus small. So we sort the `logxs`.
    #
    # Starting the summation from the *smallest* numbers should allow us to accumulate them
    # before we lose the mantissa bits to represent them due to the increasing exponent.
    # If this is really important, we could do it in pure Python, and `math.fsum` them.
    # But we have likely already lost more accuracy due to cancellation, so let's not bother
    # overengineering this part.
    logxs = tf.sort(logxs, axis=0, direction="ASCENDING")
    # # What we want to do:
    # from unpythonic import window
    # out = logxs[0]  # log(x[0])
    # for prev, curr in window(2, logxs):  # log(x[k]) - log(x[k-1])
    #     out += tf.math.softplus(curr - prev)
    sp_diffs = tf.math.softplus(logxs[1:] - logxs[:-1])  # softplus(log(x[k]) - log(x[k-1]))
    return logxs[0] + tf.reduce_sum(sp_diffs)


# --------------------------------------------------------------------------------
# The actual performance statistics

def active_units(model, x, *, batch_size=1024, eps=0.1):
    """[performance statistic] Compute AU, the number of latent active units.

    `x`: tensor of shape (N, 28, 28, 1); data batch of grayscale pictures

    Returns an int, the number of latent active units for given model,
    estimated using the given data.

    It is preferable to pass as much data as possible (e.g. all of the
    test data) to get a good estimate of AU.

    AU measures how many of the available latent dimensions a trained VAE actually uses,
    so the maximum possible value is `latent_dim`. A higher value is better.

    The log of the covariance typically has a bimodal distribution, so AU is not very
    sensitive to the value of ϵ, as long as it is between the peaks.

    We define AU as::

        AU := #{i = 1, ..., d: abs(covar(x,  E[z ~ qϕ(z|x)](zi))) > ϵ}

    where d is the dimension of the latent space, ϵ is a suitable small number,
    and #{} counts the number of elements of a set.

    See Burda et al. (2016):
       https://arxiv.org/abs/1509.00519
    """
    # As a classical numericist totally not versed in any kind of statistics / #helpwanted:
    #
    # Burda's original definition of the activation statistic AU is essentially
    #
    #   AU := #{u: Au > ϵ}
    #
    # where (and I quote literally)
    #
    #   Au := Cov_{x}( E(u ∼ qφ(u|x))[u] )
    #
    # Here u is a component of the code point vector z (in our notation, u → zi),
    # and the x is boldface, denoting the whole input picture. The authors used ϵ = 0.01.
    #
    # No one else (except other papers citing this exact definition from Burda et al.)
    # seems to use that subscript notation for covariance, not to mention using an initial
    # capital "C"; I couldn't find a definitive definition *anywhere* pinning down the
    # exact meaning of this variant of the notation.
    #
    # The text in section 5.2 of the paper implies it is indeed some kind of covariance,
    # since the use case of this activation statistic is to measure whether each u (i.e. zi)
    # affects the output of the generative model (decoder) or not.
    #
    # Two more details are missing from the paper:
    #
    #   - The input picture x, interpreted as a data vector, has  n_pixels = ny * nx * c
    #     components. The latent code z has latent_dim components. Therefore,
    #     the sample covariance between observations x and z is a matrix of size
    #     [n_pixels, latent_dim]. But the paper hints that the maximum possible value
    #     of AU is latent_dim; which implies we should reduce over the pixels, leaving
    #     only latent_dim components for the covariance (aggregate effect of each zi on
    #     the picture x). Should we sum over the pixels? Take the mean over the pixels?
    #     Something else? (We have chosen to sum.)
    #
    #   - Generally, covariance may also be negative, the sign giving the sense of
    #     the detected linear relationship. Yet in Appendix C, the authors speak of
    #     plotting its (real-valued) log, which for a negative input is clearly NaN.
    #     I think the definition must be missing an abs(), unless this is implied by
    #     the notation "Cov_{x}(...)" instead of the standard "cov(x, ...)".
    #     (We have chosen to take abs() before comparing to ϵ.)
    #
    # If I understand this right,  E[z ~ qϕ(z|x)](z) = μ  from the encoder.
    # We give the encoder the variational parameters ϕ (trained NN coefficients)
    # and an input picture x, and it gives us a multivariate gaussian with diagonal
    # covariance, parameterized by the vectors (μ, log σ), and conditioned on the
    # input x. For the given input x, the expectation of this gaussian is μ.
    #
    # To compute the covariance, we must then encode the whole dataset (that we wish
    # to use to estimate AU), and find the sample mean of this expectation, μbar.
    #
    # Covariance between two continuous random variables x and y is defined as
    #
    #   covar(x, y) := ∫ (x - xbar) (y - ybar) dp
    #                = ∫ (x - xbar) (y - ybar) p(x, y) dx dy
    #
    # But we're working with a dataset, so more directly relevant for us is the
    # sample covariance:
    #
    #   covar(x, y) := (1 / (N - 1)) ∑k (xk - xbar) (yk - ybar)
    #
    # where k indexes the observations. Note we essentially want to correlate the
    # behavior of the random variables X and Y across observations, so we need equally many
    # observations xk and yk. (Which we indeed have, since encoding one x produces one μ.)
    #
    # The -1 is Bessel's correction, accounting for the fact that the population mean
    # is unknown, so we use the sample mean, which is not independent of the samples.
    μ, ignored_logσ = model.encoder.predict(x, batch_size=batch_size)
    xbar = tf.reduce_mean(x, axis=0)  # pixelwise mean (over dataset)
    μbar = tf.reduce_mean(μ, axis=0)  # latent-dimension-wise mean (over dataset)

    # Like the scatter matrix in statistics, but summed over pixels and channels of `x`.
    @batched(batch_size)  # won't fit in VRAM on the full training dataset
    def scatter(x, μ):  # ([N, xy, nx, c], [N, xy, nx, c]) -> [N, latent_dim]
        xdiff = tf.reduce_sum((x - xbar), axis=[1, 2, 3])  # TODO: is this the right kind of reduction here?
        outs = []
        for d in range(latent_dim):  # covar(x, z_d)
            outs.append(xdiff * (μ[:, d] - μbar[d]))  # -> [batch_size]
        return tf.stack(outs, axis=-1)  # -> [batch_size, latent_dim]
    N = tf.shape(x)[0]
    sample_covar = (1. / (float(N) - 1.)) * tf.reduce_sum(scatter(x, μ), axis=0)
    return int(tf.reduce_sum(tf.where(tf.greater(tf.math.abs(sample_covar), eps), 1.0, 0.0)))


def negative_log_likelihood(model, x, *, batch_size=1024, n_mc_samples=10):
    """[performance statistic] Compute the negative log-likelihood (NLL).

    `x`: tensor of shape (N, 28, 28, 1); data batch of grayscale pictures

    Returns a float, the mean NLL for the given data, using the given model.

    When `x` is held-out data, NLL measures generalization (smaller is better).
    For a single input sample `x`, the NLL is defined as::

        log pθ(x) = -log( E[z ~ qϕ(z|x)]( pθ(x, z) / qϕ(z|x) ) )

    This expression is intractable, so we approximate it using Monte Carlo::

        log pθ(x) ≈ -log( (1/S) ∑s (pθ(x, z[s]) / qϕ(z[s]|x)) )

    where `S = n_mc_samples` and z[s] are the Monte Carlo samples of z ~ qϕ(z|x).
    The NLL is pretty much the ELBO as used in VAE training, but with some differences:

      - Multiple MC samples to improve accuracy.
      - The mean is computed over the whole dataset `x`, not over each batch.

    For numerical reasons, we accumulate the MC samples without evaluating
    `pθ(x, z[s])` directly, preferring to work on `log pθ(x, z[s])` instead.
    Consider that the joint probability can be rewritten as

        pθ(x, z) = pθ(x|z) pθ(z)

    Whereas pθ(z) and qϕ(z|x) are gaussians, with reasonable log-probabilities,
    for the continuous-Bernoulli VAE, on MNIST, `log pθ(x|z) ~ +1500` (!).
    This is technically fine, because pθ is a probability *density*, but it
    cannot be exp'd without causing overflow, even at float64.

    To overcome this, we use the smoothmax identity for the logarithm of a sum.
    Let `r[s] := pθ(x, z[s]) / qϕ(z[s]|x)`. The identity allows us to express
    `log(∑s r[s])` in terms of the `log(r[s])`::

        log(x + y) = log x + softplus(log y - log x)

    where::

        softplus(x) ≡ log(1 + exp(x))

    To obtain `log(∑s r[s])`, we start from `log r[0]`, and then (using the
    associative property of addition) accumulate over 2-tuples in a loop
    (which we vectorize for speed).

    The final detail is to handle the global scaling factor in the MC
    representation of the expectation, but this is easy::

       log(α x) = log α + log x

    so that we actually evaluate::

        log(α ∑s r[s]) = log α + log(∑s r[s])

    The definition of the NLL metric is given e.g. in Sinha and Dieng (2022):
      https://arxiv.org/pdf/2105.14859.pdf
    The smoothmax identity for the logarithm of a sum is discussed e.g. in:
      https://cdsmithus.medium.com/the-logarithm-of-a-sum-69dd76199790
    """
    print("NLL: encoding...")
    mean, logvar = model.encoder.predict(x, batch_size=batch_size)

    @batched(batch_size)
    def samplewise_elbo(x, mean, logvar):  # positional parameters get @batched
        """log(pθ(x, z) / qϕ(z|x)) for each sample of x, drawing one MC sample of z.

        `x`: tensor of shape (N, 28, 28, 1); data batch of grayscale pictures
        `mean`, `logvar`: output of encoder with input `x`

        Returns a tensor of shape (N,).
        """
        ignored_eps, z = model.reparameterize(mean, logvar)  # draw MC sample
        # Rewriting the joint probability:
        #   pθ(x, z) = pθ(x|z) pθ(z)
        # we have
        #   log pθ(x, z) = log(pθ(x|z) pθ(z))
        #                = log pθ(x|z) + log pθ(z)
        logpx_z = log_px_z(model, x, z, training=False)  # log pθ(x|z)
        logpz = log_normal_pdf(z, 0., 0.)                # log pθ(z)
        logpxz = logpx_z + logpz                         # log pθ(x, z)

        logqz_x = log_normal_pdf(z, mean, logvar)        # log qϕ(z|x)

        return logpxz - logqz_x

    print(f"NLL: MC sampling (n = {n_mc_samples})...")
    acc = [samplewise_elbo(x, mean, logvar) for _ in range(n_mc_samples)]  # -> [[N], [N], ...]
    acc = tf.stack(acc, axis=1)  # -> [N, n_mc_samples]

    # TODO: I think this is correct, we should reduce linearly here. But check just to be sure.
    # Taking the mean like this computes (albeit averaging over `x` too early; strictly, we should accumulate the MC samples first):
    #   -E[x ~ data](log E[z ~ qϕ(z|x)]( pθ(x, z) / qϕ(z|x) ))
    # whereas treating both dimensions the same (flatten, send to accumulation loop) would compute:
    #   -log E[x ~ data](E[z ~ qϕ(z|x)]( pθ(x, z) / qϕ(z|x) ))
    acc = tf.reduce_mean(acc, axis=0)  # -> [n_mc_samples]

    print("NLL: computing MC estimate...")
    # `log(∑k r[k])`, in terms of `log r[k]`, using the smoothmax identity.
    out = logsumxs(acc)
    out += tf.math.log(1. / float(tf.shape(acc)))  # scaling in the expectation operator
    return -out.numpy()  # *negative* log-likelihood


def mutual_information(model, x, *, batch_size=1024, n_mc_samples=10):
    """[performance statistic] Compute mutual information between x and its code z.

    We actually compute and return two related metrics; the KL regularization
    term of the ELBO, namely the KL divergence of the variational posterior
    from the latent prior::

        E[x ~ pd(x)]( DKL[qϕ(z|x) ‖ pθ(z)] )

    and the mutual information induced by the variational joint, defined as::

        I[q](x, z) := E[x ~ pd(x)]( DKL[qϕ(z|x) ‖ pθ(z)] - DKL[qϕ(z) ‖ pθ(z)] )

    where DKL is the Kullback-Leibler divergence::

        DKL[q(z) ‖ p(z)] ≡ E[z ~ q(z)]( log q(z) - log p(z) ),

    pd(x) is the empirical data distribution, and qϕ(z) is the aggregated posterior,
    which is the marginal over z induced by the joint::

        qϕ(z, x) := qϕ(z|x) pd(x)

    or in other words,

        qφ(z) ≡ ∫ qφ(z|x) pd(x) dx

    The return value is `(DKL, MI)`, approximated using Monte Carlo.

    Slightly different definitions the MI metric are given e.g. in
    Sinha and Dieng (2022), and in Dieng et al. (2019):
      https://arxiv.org/pdf/2105.14859.pdf
      https://arxiv.org/pdf/1807.04863.pdf
    """
    # TODO: Refactor this, useful as-is (the "KL" metric reported in some papers).
    #
    # TODO: Investigate: if qϕ(z|x) and pθ(z) are both gaussian (as they are here), Takahashi et al. (2019)
    #       note that it should be possible to evaluate the KL divergence in closed form, citing the
    #       original VAE paper by Kingma and Welling (2013):
    #       [Kingma and Welling 2013] Kingma, D. P., and Welling, M.2013. Auto-encoding variational Bayes.
    #       arXiv preprint arXiv:1312.6114.
    def dkl_qz_x_from_pz(x, n_mc_samples):
        """Estimate the KL divergence of the variational posterior from the latent prior.

        This is the KL regularization term that appears in the ELBO
        (in one of its alternative expressions).

        Defined as::
            DKL(qϕ(z|x) ‖ pθ(z)) ≡ E[z ~ qϕ(z|x)]( log qϕ(z|x) - log pθ(z) )

        `n_z_mc_samples` is the number of Monte Carlo samples to use to
        estimate the expectation.
        """
        mean, logvar = model.encoder.predict(x, batch_size=batch_size)  # prevent re-batching into smaller subbatches
        @batched(batch_size)
        def logratio(x, mean, logvar):  # positional parameters get @batched
            # Given an `x`, we draw a single MC sample of `z ~ qϕ(z|x)`, where the distribution is given by the encoder.
            ignored_eps, z = model.reparameterize(mean, logvar)
            logqz_x = log_normal_pdf(z, mean, logvar)  # log qϕ(z|x)
            logpz = log_normal_pdf(z, 0., 0.)          # log pθ(z)
            return logqz_x - logpz

        # E[z ~ qϕ(z|x)](...)
        out = [logratio(x, mean, logvar) for _ in range(n_mc_samples)]  # [N, n_mc_samples]
        out = tf.reduce_mean(out, axis=1)  # [N]
        return out

    # The first DKL term. For each given `x`:
    #   DKL(qϕ(z|x) ‖ pθ(z)) ≡ E[z ~ qϕ(z|x)]( log qϕ(z|x) - log pθ(z) )
    first_dkl_term = dkl_qz_x_from_pz(x, n_mc_samples)  # [N]

    # The second DKL term (note this does not depend on `x`):
    #   DKL(qϕ(z) ‖ pθ(z)) = E[z ~ qϕ(z)]( log qϕ(z) - log pθ(z) ),
    #
    # For this, we need access to the aggregated posterior qϕ(z). Ouch!
    #
    # If a joint distribution is available, it is possible to obtain samples from a marginal by sampling the joint,
    # by just ignoring the other outputs:
    #   https://math.stackexchange.com/questions/3236982/marginalizing-by-sampling-from-the-joint-distribution
    #
    # But even better here is:
    #
    # "We can sample from p(z) and qφ(z|x) since these distributions are a Gaussian, and we can also sample from
    # the aggregated posterior qφ(z) by using ancestral sampling: we choose a data point x from a dataset randomly
    # and sample z from the encoder given this data point x."
    #   --Takahashi et al. (2019):
    #     https://arxiv.org/pdf/1809.05284.pdf
    #
    # That gives us a single Monte Carlo sample of `z ~ qφ(z)`, which we can plug into `log qφ(z)` (and thus into
    # the second DKL term). To get the expectation, we average this MC sampling over `z` the usual way.
    #
    # For each sample `z`, how to actually evaluate the log-density `log qφ(z)`? The aggregated posterior is:
    #
    #   qφ(z) ≡ ∫ qφ(z|x) pd(x) dx      (marginalize away `x` in the joint `qϕ(z, x) = qϕ(z|x) pd(x)`)
    #         = E[x ~ pd(x)](qφ(z|x))
    #         ≈ (1/K) ∑k qφ(z|xk)       (Monte Carlo estimate)
    #
    # where, since `x ~ pd(x)`, we just draw K samples `xk` from the dataset. Taking the `log`,
    #
    #   log qφ(z) ≈ log( (1/K) ∑k qφ(z|xk) )
    #             = log(1/K) + log(∑k qφ(z|xk))
    #
    # We can use the smoothmax identity to evaluate the logarithm of the sum in terms of `log qφ(z|xk)`.
    # We evaluate each `log qφ(z|xk)` at the already sampled values `z ~ qφ(z)` and `xk ~ pd(x)`.
    # In practice:
    #    - Use the already sampled `z`
    #    - Use (μ, log σ) of the model corresponding to each sampled `xk`
    #
    # TODO: refactor; there's a lot of useful stuff here (e.g. `compute_logqz` is useful on its own, to plot the aggregated posterior log-density).
    # TODO: when using the whole dataset for MC samples, we could precompute `mean` and `logvar` in one go for *all* `x`.
    def dkl_qz_from_pz(n_mc_samples):
        """Estimate the KL divergence of the aggregated posterior from the latent prior.

        This term appears in the mutual information (MI).

        Defined as::
            DKL(qϕ(z) ‖ pθ(z)) ≡ E[z ~ qϕ(z)]( log qϕ(z) - log pθ(z) )

        `n_z_mc_samples` is the number of Monte Carlo samples to use to
        estimate the expectation.
        """
        def sample_x(n):
            """Sample `x ~ pd(x)`, i.e. draw random samples from the dataset.

            `n`: how many `x` to return (as tensor of shape (n,))
            """
            # https://stackoverflow.com/questions/50673363/in-tensorflow-randomly-subsample-k-entries-from-a-tensor-along-0-axis
            ks = tf.range(tf.shape(x)[0])
            random_ks = tf.random.shuffle(ks)[:n]
            return tf.gather(x, random_ks)

        def sample_qz(n):
            """Ancestrally sample `z ~ qφ(z)`.

            `n`: how many `z` to return (as tensor of shape (n,))
            """
            # Step 1: randomly pick `n` samples from the dataset.
            xs = sample_x(n)
            # Step 2: sample one `z` for each `x`.
            mean, logvar = model.encoder.predict(xs, batch_size=batch_size)
            zs = model.reparameterize(mean, logvar)
            return zs

        # TODO: vectorize for many `z` at once
        def compute_logqz(z, n_mc_samples):
            """Evaluate aggregated posterior qφ(z) at `z`, using MC sampling."""
            xk = sample_x(n_mc_samples)  # inner MC sample: for evaluation of qφ(z)

            # For each sample `xk`, evaluate `log qϕ(z|xk)`.
            # For this we need the variational posterior parameters, so encode `xk`:
            mean, logvar = model.encoder.predict(xk, batch_size=batch_size)  # each of mean, logvar: [n_mc_samples, latent_dim]

            # z_broadcast = tf.expand_dims(z, axis=0)
            logqz_x = log_normal_pdf(z, mean, logvar)  # log qϕ(z|x)  # [n_mc_samples]

            # `log(∑k qφ(z|xk))`, in terms of `log qφ(z|xk)`, using the smoothmax identity.
            logqz = logsumxs(logqz_x)
            logqz += tf.math.log(1. / float(tf.shape(logqz_x)))  # scaling in the expectation operator
            return logqz

        # Compute the DKL, and average over the MC samples `(z, log qφ(z))`.
        # With the above definitions, this is as simple as:
        dkls = []
        for z in sample_qz(n_mc_samples):  # outer MC sample: for ancestral sampling of `z`
            logqz = compute_logqz(z, n_mc_samples)
            logpz = log_normal_pdf(z, 0., 0.)
            dkls.append(logqz - logpz)
        dkl = tf.reduce_mean(dkls).numpy()  # evaluate the MC expectation
        return dkl

    second_dkl_term = dkl_qz_from_pz(n_mc_samples)  # just a scalar

    MI = first_dkl_term - second_dkl_term

    # Finally, average over the dataset `x`.
    dkl = tf.reduce_mean(first_dkl_term).numpy()
    MI = tf.reduce_mean(MI).numpy()
    return dkl, MI
