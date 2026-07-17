# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import torch

from .utils import deg2rad, make_grid, rotmat2d
from .voting import argmax_xyr, log_softmax_spatial, sample_xyr


def local_expectation_xyr(belief, radius=3):
    """Sub-bin xy readout: argmax picks the peak, the expectation over a
    small window around it recovers the sub-cell position. The window
    excludes the other modes and the recovery floor, so the local mean is
    taken over a single unimodal bump, where it is the optimal readout.
    Yaw is left at the argmax bin. radius=0 falls back to pure argmax.
    """
    xyr = argmax_xyr(belief).to(belief)
    if radius > 0:
        h, w = belief.shape[:2]
        x, y = int(xyr[0]), int(xyr[1])
        x0, x1 = max(x - radius, 0), min(x + radius + 1, w)
        y0, y1 = max(y - radius, 0), min(y + radius + 1, h)
        patch = belief[y0:y1, x0:x1].logsumexp(-1)  # spatial log-marginal
        prob = (patch - patch.max()).exp()
        prob = prob / prob.sum()
        dx = torch.arange(x0, x1, device=belief.device, dtype=prob.dtype) - xyr[0]
        dy = torch.arange(y0, y1, device=belief.device, dtype=prob.dtype) - xyr[1]
        grid = torch.stack(torch.meshgrid(dx, dy, indexing="xy"), -1)
        xyr = xyr.clone()
        xyr[:2] += torch.einsum("hw,hwd->d", prob, grid)
    return xyr


def propagate_belief(
    Δ_xy, Δ_yaw, canvas_target, canvas_source, belief, num_rotations=None
):
    # We allow a different sampling resolution in the target frame
    if num_rotations is None:
        num_rotations = belief.shape[-1]

    angles = torch.arange(
        0, 360, 360 / num_rotations, device=Δ_xy.device, dtype=Δ_xy.dtype
    )
    uv_grid = make_grid(canvas_target.w, canvas_target.h, device=Δ_xy.device)
    xy_grid = canvas_target.to_xy(uv_grid.to(Δ_xy))

    Δ_xy_world = torch.einsum("nij,j->ni", rotmat2d(deg2rad(-angles)), Δ_xy)
    xy_grid_prev = xy_grid[..., None, :] + Δ_xy_world[..., None, None, :, :]
    uv_grid_prev = canvas_source.to_uv(xy_grid_prev).to(Δ_xy)

    angles_prev = angles + Δ_yaw
    angles_grid_prev = angles_prev.tile((canvas_target.h, canvas_target.w, 1))

    prior, valid = sample_xyr(
        belief[None, None],
        uv_grid_prev.to(belief)[None],
        angles_grid_prev.to(belief)[None],
        nearest_for_inf=True,
    )
    return prior, valid


def observation_weight(log_probs, gamma=1.0):
    """Confidence of an observation in [0, 1] from its normalized entropy.

    A peaked observation (low entropy) gets w close to 1, an ambiguous one
    (entropy close to the uniform maximum) gets w close to 0. Scaling the
    observation log-likelihood by w tempers it, so poor observations vote
    less during the fusion.
    """
    probs = log_probs.exp()
    entropy = -torch.xlogy(probs, probs).sum()
    certainty = 1 - entropy / np.log(log_probs.numel())
    return certainty.clamp(min=1e-3, max=1.0) ** gamma


def integrate_observation(
    source,
    target,
    xy_source,
    xy_target,
    yaw_source,
    yaw_target,
    canvas_source,
    canvas_target,
    **kwargs
):
    Δ_xy = rotmat2d(deg2rad(yaw_target)) @ (xy_source - xy_target)
    Δ_yaw = yaw_source - yaw_target
    prior, valid = propagate_belief(
        Δ_xy, Δ_yaw, canvas_target, canvas_source, source, **kwargs
    )
    # Hypotheses outside the previous canvas fall back to a uniform
    # prior (not -inf), so they remain recoverable.
    uniform = -np.log(prior[0, 0].numel())
    prior = prior[0, 0].masked_fill_(~valid[0], uniform)
    target.add_(prior)
    target.sub_(target.max())  # normalize to avoid overflow
    return prior


class RigidAligner:
    def __init__(self):
        self.canvas = None
        self.xy_ref = None
        self.yaw_ref = None
        self.belief = None

    def update_with_ref(self, observation, canvas, xy, yaw, recovery_prior, recovery_eps=0.01, obs_weight_gamma=1.0):
        """recovery_prior: (h, w) log-prior (e.g. the OSM road prior) shaping
        where the ε recovery mass goes."""
        if obs_weight_gamma > 0:
            # Entropy-based confidence: ambiguous observations vote less.
            w = observation_weight(observation, obs_weight_gamma)
            observation = observation * w
        if self.belief is not None:
            observation = observation.clone()
            integrate_observation(
                self.belief,
                observation,
                self.xy_ref,
                xy,
                self.yaw_ref,
                yaw,
                self.canvas,
                canvas,
                num_rotations=observation.shape[-1],
            )

        belief = log_softmax_spatial(observation)
        if recovery_eps > 0:
            # Recovery mass shaped by the semantic prior: normalized over
            # space, uniform over rotations, blended with 10% uniform so
            # that every pose stays recoverable.
            prior = recovery_prior.to(belief)
            prior = prior - torch.logsumexp(prior.flatten(), 0)
            prior = prior[..., None] - np.log(belief.shape[-1])
            recovery = torch.logaddexp(
                prior + np.log(0.9),
                prior.new_full((), np.log(0.1) - np.log(belief.numel())),
            )
            belief = torch.logaddexp(
                belief + np.log1p(-recovery_eps),
                np.log(recovery_eps) + recovery,
            )
        self.belief = belief
        self.canvas = canvas
        self.xy_ref = xy
        self.yaw_ref = yaw
