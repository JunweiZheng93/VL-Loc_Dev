f_v5_rot = self._rotate_bev_around_camera(f_v5, 30)
v_v5_rot = (
    self._rotate_bev_around_camera(
        v_v5.unsqueeze(0).float(), 30
    )
    .squeeze(0) > 0.5
)

f_v6_rot = self._rotate_bev_around_camera(f_v6, -30)
v_v6_rot = (
    self._rotate_bev_around_camera(
        v_v6.unsqueeze(0).float(), -30
    )
    .squeeze(0) > 0.5
)

f_v5_masked = torch.where(v_v5_rot.unsqueeze(0), f_v1_rot, 0.0)
f_v6_masked = torch.where(v_v6_rot.unsqueeze(0), f_v2_rot, 0.0)

numerator_V3 = (f_v5_masked * v_v5_rot.unsqueeze(0).float() +
            f_v6_masked * v_v6_rot.unsqueeze(0).float())

denominator_V3 = (v_v5_rot.float() + v_v6_rot.float())
denominator_V3 = torch.clamp(denominator, min=0.5)

f_bev_stitched_V3[i] = numerator_V3 / denominator_V3.unsqueeze(0)
valid_stitched_V3[i] = v_v5_rot | v_v6_rot