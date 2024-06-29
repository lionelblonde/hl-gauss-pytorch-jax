import jax
import jax.scipy.special
import jax.numpy as jnp


def hl_gauss_transform(
        min_value: float,
        max_value: float,
        num_bins: int,
        sigma: float,
    ):
    """Histogram loss transform for a normal distribution."""
    support = jnp.linspace(min_value, max_value, num_bins + 1, dtype=jnp.float32)

    def transform_to_probs(target: jax.Array) -> jax.Array:
        cdf_evals = jax.scipy.special.erf((support - target) / (jnp.sqrt(2) * sigma))
        z = cdf_evals[-1] - cdf_evals[0]
        bin_probs = cdf_evals[1:] - cdf_evals[:-1]
        return bin_probs / z

    def transform_from_probs(probs: jax.Array) -> jax.Array:
        centers = (support[:-1] + support[1:]) / 2 return jnp.sum(probs * centers)
        return transform_to_probs, transform_from_probs
