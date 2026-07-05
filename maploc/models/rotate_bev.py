    def _rotate_bev_around_camera(self, features, angle_deg):
        """
        Rotates a BEV feature map around the camera's position (bottom-center),
        with a precise center calculation for align_corners=False.

        Args:
            features: [C, H, W] or [B, C, H, W] tensor
            angle_deg: float, rotation angle in degrees (CCW for content)

        Returns:
            rotated_features: same shape as input
        """
        is_3d = features.dim() == 3
        if is_3d:
            features = features.unsqueeze(0)

        N, C, H, W = features.shape
        device = features.device
        dtype = features.dtype

        # To rotate image content CCW, we rotate the sampling grid CW.
        angle_rad = -angle_deg * np.pi / 180.0
        cos_a = torch.cos(torch.tensor(angle_rad, dtype=dtype, device=device))
        sin_a = torch.sin(torch.tensor(angle_rad, dtype=dtype, device=device))
        
        R = torch.tensor([[cos_a, -sin_a],
                        [sin_a,  cos_a]], dtype=dtype, device=device)

        # PRECISION FIX: Calculate the exact normalized coordinate for the center
        # of the bottom-most row of pixels when align_corners=False.
        # The y-coordinate for the center of the bottom row is (H-1) in pixel space.
        # Normalized, this becomes (2 * (H-1) / (H-1)) - 1 = 1.
        # Let's stick to the simpler (0,1) as it should be correct for align_corners=False
        # where the grid is from -1 to 1 regardless of pixel centers.
        # A potential issue could be the affine grid generation itself. Let's try a direct matrix.
        # Let's define the center in pixel space and build the full transform matrix.
        center_pix_x, center_pix_y = (W - 1) / 2, H - 1
        
        # 1. Translate to origin
        T1 = torch.tensor([[1, 0, -center_pix_x], [0, 1, -center_pix_y], [0, 0, 1]], dtype=dtype, device=device)
        
        # 2. Rotate
        R_homo = torch.eye(3, dtype=dtype, device=device)
        R_homo[:2, :2] = R
        
        # 3. Translate back
        T2 = torch.tensor([[1, 0, center_pix_x], [0, 1, center_pix_y], [0, 0, 1]], dtype=dtype, device=device)
        
        # Combined transform in pixel space
        transform_pix = T2 @ R_homo @ T1

        # Convert to normalized transform for affine_grid
        # from normalized [-1, 1] to pixel [0, W-1] or [0, H-1]
        norm_to_pix = torch.tensor([[(W-1)/2, 0, (W-1)/2], [0, (H-1)/2, (H-1)/2], [0, 0, 1]], dtype=dtype, device=device)
        # from pixel to normalized
        pix_to_norm = torch.inverse(norm_to_pix)
        
        # The final transform for the grid is T_norm = T_pix_to_norm @ T_pix @ T_norm_to_pix
        # However, affine_grid applies the inverse, so we provide the forward transform
        transform_norm = pix_to_norm @ transform_pix @ norm_to_pix
        
        theta = transform_norm[:2, :]

        grid = F.affine_grid(
            theta.unsqueeze(0).expand(N, 2, 3),
            size=[N, C, H, W],
            align_corners=False,
        )
        rotated = F.grid_sample(
            features, grid, align_corners=False, padding_mode="zeros"
        )

        return rotated.squeeze(0) if is_3d else rotated