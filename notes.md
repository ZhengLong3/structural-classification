# Notes from experimenting with gaussian splats

### Splat properties

- **_xyz**: (n, 3) location of the center of the splat. x seems to be to the right, y is down and z is into the screen.
- **_scaling**: (n, 3) x, y, z scale of the splat, seems to be in $\log_2$ scale.
- **_rotation**: (n, 4) quarternion signifying the rotation of the splat about its center.
- **_opacity**: (n, 1) opacity of each splat at the center. A multiplier on the distribution of the gaussian for opacity
- **_features_dc**: (n, 1, 3) RGB colour of the splat, if the features_rest is zero, as features_rest affect the colour through spherical harmonical encoding
- **_features_rest**: (n, 15, 3) affects the colour of the splat when viewed from different directions. Refer to [this](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/sh_utils.py) for computation of the colour given direction and features_dc and features_rest.