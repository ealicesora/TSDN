import torch
from torch.func import jacfwd

def log1p_safe(x):
    """The same as tf.math.log1p(x), but clamps the input to prevent NaNs."""
    return torch.log1p(torch.minimum(x, torch.tensor(3e37)))


def exp_safe(x):
    """The same as tf.math.exp(x), but clamps the input to prevent NaNs."""
    return torch.exp(torch.minimum(x, torch.tensor(87.5)))


def expm1_safe(x):
    """The same as tf.math.expm1(x), but clamps the input to prevent NaNs."""
    return torch.expm1(torch.minimum(x, torch.tensor(87.5)))

def general_loss_with_squared_residual(squared_x, alpha, scale):
    r"""The general loss that takes a squared residual.

    This fuses the sqrt operation done to compute many residuals while preserving
    the square in the loss formulation.

    This implements the rho(x, \alpha, c) function described in "A General and
    Adaptive Robust Loss Function", Jonathan T. Barron,
    https://arxiv.org/abs/1701.03077.

    Args:
    squared_x: The residual for which the loss is being computed. x can have
        any shape, and alpha and scale will be broadcasted to match x's shape if
        necessary.
    alpha: The shape parameter of the loss (\alpha in the paper), where more
        negative values produce a loss with more robust behavior (outliers "cost"
        less), and more positive values produce a loss with less robust behavior
        (outliers are penalized more heavily). Alpha can be any value in
        [-infinity, infinity], but the gradient of the loss with respect to alpha
        is 0 at -infinity, infinity, 0, and 2. Varying alpha allows for smooth
        interpolation between several discrete robust losses:
        alpha=-Infinity: Welsch/Leclerc Loss.
        alpha=-2: Geman-McClure loss.
        alpha=0: Cauchy/Lortentzian loss.
        alpha=1: Charbonnier/pseudo-Huber loss.
        alpha=2: L2 loss.
    scale: The scale parameter of the loss. When |x| < scale, the loss is an
        L2-like quadratic bowl, and when |x| > scale the loss function takes on a
        different shape according to alpha.

    Returns:
    The losses for each element of x, in the same shape as x.
    """
    eps = torch.tensor(torch.finfo(torch.float32).eps)

    # This will be used repeatedly.
    squared_scaled_x = squared_x / (scale ** 2)

    # The loss when alpha == 2.
    loss_two = 0.5 * squared_scaled_x
    # The loss when alpha == 0.
    loss_zero = log1p_safe(0.5 * squared_scaled_x)
    # The loss when alpha == -infinity.
    loss_neginf = -torch.expm1(-0.5 * squared_scaled_x)
    # The loss when alpha == +infinity.
    loss_posinf = expm1_safe(0.5 * squared_scaled_x)

    # The loss when not in one of the above special cases.
    # Clamp |2-alpha| to be >= machine epsilon so that it's safe to divide by.
    beta_safe = torch.maximum(eps, torch.abs((alpha - 2.)))
    # Clamp |alpha| to be >= machine epsilon so that it's safe to divide by.
    alpha_safe = torch.where(
        torch.greater_equal(alpha, 0.), torch.ones_like(alpha),
        -torch.ones_like(alpha)) * torch.maximum(eps, torch.abs(alpha))
    loss_otherwise = (beta_safe / alpha_safe) * (
        torch.pow(squared_scaled_x / beta_safe + 1., 0.5 * alpha) - 1.)

    # Select which of the cases of the loss to return.
    loss = torch.where(
        alpha == -torch.inf, loss_neginf,
        torch.where(
            alpha == 0, loss_zero,
            torch.where(
                alpha == 2, loss_two,
                torch.where(alpha == torch.inf, loss_posinf, loss_otherwise))))

    return scale * loss


def compute_elastic_loss_fancy(jacobian, eps=1e-6, loss_type='log_svals'):
    """Compute the elastic regularization loss.

    The loss is given by sum(log(S)^2). This penalizes the singular values
    when they deviate from the identity since log(1) = 0.0,
    where D is the diagonal matrix containing the singular values.

    Args:
    jacobian: the Jacobian of the point transformation.
    eps: a small value to prevent taking the log of zero.
    loss_type: which elastic loss type to use.

    Returns:
    The elastic regularization loss.
    """
    eps = torch.tensor(eps)
    svals = torch.linalg.svdvals(jacobian)
    log_svals = torch.log(torch.maximum(svals, eps))
    sq_residual = torch.sum(log_svals**2, axis=-1)
    residual = torch.sqrt(sq_residual)
    loss = general_loss_with_squared_residual(
    sq_residual, alpha=torch.tensor( -2.0), scale=0.03)
    return loss, residual

def compute_elastic_loss(jacobian, eps=1e-6, loss_type='log_svals'):
    """Compute the elastic regularization loss.

    The loss is given by sum(log(S)^2). This penalizes the singular values
    when they deviate from the identity since log(1) = 0.0,
    where D is the diagonal matrix containing the singular values.

    Args:
    jacobian: the Jacobian of the point transformation.
    eps: a small value to prevent taking the log of zero.
    loss_type: which elastic loss type to use.

    Returns:
    The elastic regularization loss.
    """
    eps = torch.tensor(eps)
    svals = torch.linalg.svdvals(jacobian)
    # print(svals.shape)
    log_svals = torch.maximum(svals, eps)
    sq_residual = torch.sum(log_svals**2, axis=-1)
    residual = torch.sqrt(sq_residual)
    loss = general_loss_with_squared_residual(
    sq_residual, alpha=torch.tensor( -2.0), scale=0.03)
    return loss, residual


def compute_divergence_loss(jacobian):
    #print(jacobian.shape)
    diagonal = jacobian.reshape(jacobian.shape[0], -1)[:, :: (jacobian.shape[1]+1)]
    sums = torch.sum(diagonal - 1.0, 1)
    divergence_loss = torch.abs(sums)
    divergence_loss = divergence_loss ** 2
    #print(divergence_loss.shape)
    return divergence_loss,0.0


def grid_sample_3d_customize(image, optical):
    N, C, ID, IH, IW = image.shape
    _, D, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]
    iz = optical[..., 2]

    ix = ((ix + 1) / 2) * (IW - 1);
    iy = ((iy + 1) / 2) * (IH - 1);
    iz = ((iz + 1) / 2) * (ID - 1);
    with torch.no_grad():
        
        ix_tnw = torch.floor(ix);
        iy_tnw = torch.floor(iy);
        iz_tnw = torch.floor(iz);

        ix_tne = ix_tnw + 1;
        iy_tne = iy_tnw;
        iz_tne = iz_tnw;

        ix_tsw = ix_tnw;
        iy_tsw = iy_tnw + 1;
        iz_tsw = iz_tnw;

        ix_tse = ix_tnw + 1;
        iy_tse = iy_tnw + 1;
        iz_tse = iz_tnw;

        ix_bnw = ix_tnw;
        iy_bnw = iy_tnw;
        iz_bnw = iz_tnw + 1;

        ix_bne = ix_tnw + 1;
        iy_bne = iy_tnw;
        iz_bne = iz_tnw + 1;

        ix_bsw = ix_tnw;
        iy_bsw = iy_tnw + 1;
        iz_bsw = iz_tnw + 1;

        ix_bse = ix_tnw + 1;
        iy_bse = iy_tnw + 1;
        iz_bse = iz_tnw + 1;

    tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
    tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
    tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
    tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
    bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
    bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
    bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
    bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);

    with torch.no_grad():

        ix_tnw = torch.clamp(ix_tnw, 0, IW - 1)
        iy_tnw = torch.clamp(iy_tnw, 0, IH - 1)
        iz_tnw = torch.clamp(iz_tnw, 0, ID - 1)

        ix_tne = torch.clamp(ix_tne, 0, IW - 1)
        iy_tne = torch.clamp(iy_tne, 0, IH - 1)
        iz_tne = torch.clamp(iz_tne, 0, ID - 1)

        ix_tsw = torch.clamp(ix_tsw, 0, IW - 1)
        iy_tsw = torch.clamp(iy_tsw, 0, IH - 1)
        iz_tsw = torch.clamp(iz_tsw, 0, ID - 1)

        ix_tse = torch.clamp(ix_tse, 0, IW - 1)
        iy_tse = torch.clamp(iy_tse, 0, IH - 1)
        iz_tse = torch.clamp(iz_tse, 0, ID - 1)

        ix_bnw = torch.clamp(ix_bnw, 0, IW - 1)
        iy_bnw = torch.clamp(iy_bnw, 0, IH - 1)
        iz_bnw = torch.clamp(iz_bnw, 0, ID - 1)

        ix_bne = torch.clamp(ix_bne, 0, IW - 1)
        iy_bne = torch.clamp(iy_bne, 0, IH - 1)
        iz_bne = torch.clamp(iz_bne, 0, ID - 1)

        ix_bsw = torch.clamp(ix_bsw, 0, IW - 1)
        iy_bsw = torch.clamp(iy_bsw, 0, IH - 1)
        iz_bsw = torch.clamp(iz_bsw, 0, ID - 1)

        ix_bse = torch.clamp(ix_bse, 0, IW - 1)
        iy_bse = torch.clamp(iy_bse, 0, IH - 1)
        iz_bse = torch.clamp(iz_bse, 0, ID - 1)

    # with torch.no_grad():

    #     torch.clamp(ix_tnw, 0, IW - 1, out=ix_tnw)
    #     torch.clamp(iy_tnw, 0, IH - 1, out=iy_tnw)
    #     torch.clamp(iz_tnw, 0, ID - 1, out=iz_tnw)

    #     torch.clamp(ix_tne, 0, IW - 1, out=ix_tne)
    #     torch.clamp(iy_tne, 0, IH - 1, out=iy_tne)
    #     torch.clamp(iz_tne, 0, ID - 1, out=iz_tne)

    #     torch.clamp(ix_tsw, 0, IW - 1, out=ix_tsw)
    #     torch.clamp(iy_tsw, 0, IH - 1, out=iy_tsw)
    #     torch.clamp(iz_tsw, 0, ID - 1, out=iz_tsw)

    #     torch.clamp(ix_tse, 0, IW - 1, out=ix_tse)
    #     torch.clamp(iy_tse, 0, IH - 1, out=iy_tse)
    #     torch.clamp(iz_tse, 0, ID - 1, out=iz_tse)

    #     torch.clamp(ix_bnw, 0, IW - 1, out=ix_bnw)
    #     torch.clamp(iy_bnw, 0, IH - 1, out=iy_bnw)
    #     torch.clamp(iz_bnw, 0, ID - 1, out=iz_bnw)

    #     torch.clamp(ix_bne, 0, IW - 1, out=ix_bne)
    #     torch.clamp(iy_bne, 0, IH - 1, out=iy_bne)
    #     torch.clamp(iz_bne, 0, ID - 1, out=iz_bne)

    #     torch.clamp(ix_bsw, 0, IW - 1, out=ix_bsw)
    #     torch.clamp(iy_bsw, 0, IH - 1, out=iy_bsw)
    #     torch.clamp(iz_bsw, 0, ID - 1, out=iz_bsw)

    #     torch.clamp(ix_bse, 0, IW - 1, out=ix_bse)
    #     torch.clamp(iy_bse, 0, IH - 1, out=iy_bse)
    #     torch.clamp(iz_bse, 0, ID - 1, out=iz_bse)

    image = image.view(N, C, ID * IH * IW)

    tnw_val = torch.gather(image, 2, (iz_tnw * IW * IH + iy_tnw * IW + ix_tnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tne_val = torch.gather(image, 2, (iz_tne * IW * IH + iy_tne * IW + ix_tne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tsw_val = torch.gather(image, 2, (iz_tsw * IW * IH + iy_tsw * IW + ix_tsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tse_val = torch.gather(image, 2, (iz_tse * IW * IH + iy_tse * IW + ix_tse).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bnw_val = torch.gather(image, 2, (iz_bnw * IW * IH + iy_bnw * IW + ix_bnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bne_val = torch.gather(image, 2, (iz_bne * IW * IH + iy_bne * IW + ix_bne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bsw_val = torch.gather(image, 2, (iz_bsw * IW * IH + iy_bsw * IW + ix_bsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bse_val = torch.gather(image, 2, (iz_bse * IW * IH + iy_bse * IW + ix_bse).long().view(N, 1, D * H * W).repeat(1, C, 1))

    out_val = (tnw_val.view(N, C, D, H, W) * tnw.view(N, 1, D, H, W) +
               tne_val.view(N, C, D, H, W) * tne.view(N, 1, D, H, W) +
               tsw_val.view(N, C, D, H, W) * tsw.view(N, 1, D, H, W) +
               tse_val.view(N, C, D, H, W) * tse.view(N, 1, D, H, W) +
               bnw_val.view(N, C, D, H, W) * bnw.view(N, 1, D, H, W) +
               bne_val.view(N, C, D, H, W) * bne.view(N, 1, D, H, W) +
               bsw_val.view(N, C, D, H, W) * bsw.view(N, 1, D, H, W) +
               bse_val.view(N, C, D, H, W) * bse.view(N, 1, D, H, W))

    return out_val


