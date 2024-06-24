import gin
from scipy import stats
import torch


@gin.configurable()
def mask_gen(batch_size: int, image_size: int, mu: float = 1.0, sigma: float = 0.01) -> torch.Tensor:
    """Generates a random binary mask tensor using the truncated standard normal distribution.

    Args:
        batch_size: The batch size of input tensor.
        image_size: The size(height & width) of the generated images.
        mu: Mean of standard normal distribution.
        sigma: Standard deviation of standard normal distribution.

    Returns:
        torch.Tensor: A binary mask tensor of shape (batch_size, 1, image_size/4, image_size/4).
    """

    x = stats.truncnorm((0 - mu) / sigma, (1 - mu) / sigma, loc=mu, scale=sigma)
    mask_size = image_size // 4  # hint image should be 1/4 size of the input image

    mask = torch.cat([torch.rand(1, 1, mask_size, mask_size).ge(x.rvs(1)[0]) for _ in range(batch_size)], 0)
    return mask
