# UV vehicle model

The vehicle state is `[x, vx, y, vy, z, vz, yaw, yaw_rate, r1, r2-r1,
height_offset, car_pitch]`. Indices 0 through 10 retain their previous meanings.
The vehicle rotation is `Rz(yaw) * Ry(car_pitch)`: roll is fixed at zero.
The same rotation transforms armor center offsets and armor orientations.
Each armor has a fixed pitch of +15 degrees relative to the vehicle; the outpost
retains its -15 degree mounting angle. Outpost and base vehicle pitch is zero.

`Solver::solve` initializes from IPPE, then minimizes the sum of squared errors
of the four detected corners in distorted pixel coordinates using Ceres. It
jointly estimates position, yaw and pitch with roll fixed at zero; outpost/base
pitch is fixed. Camera-frame fields and `yaw_raw` retain the original PnP
result for comparison. World/gimbal fields contain the optimized result.

`Target::update(armor, solver)` associates armor IDs in the image plane and
updates the vehicle with both lights together. Each light contributes
`[angle, center_u, center_v, length]`, giving an eight-dimensional observation.
The observation uses the original detected pixels, not fitted/reprojected
corners or PnP depth. The projection includes camera distortion and the current
IMU transform; Ceres automatic differentiation supplies the state Jacobian.
Use `Target::armor_pose(id)` with the pose overload of `reproject_armor` to
draw the complete tilted armor pose. The old XYZ/yaw overload assumes a level
vehicle and is retained for callers that only have an aiming point.

Optional YAML parameters (defaults work with existing configuration files):

| Parameter | Default | Meaning |
| --- | --- | --- |
| `car_pitch_process_noise` | `0.0001` | Vehicle pitch variance growth, rad^2/s |
| `uv_sigma_center_ratio` | `0.2` | Center standard deviation / light length |
| `uv_sigma_length_ratio` | `0.3` | Length standard deviation / light length |
| `uv_sigma_angle` | `0.1` | Light angle standard deviation, rad |

As in awakening, measurement variances are half the squared standard deviations.
Center/length noise has a 0.5/1 pixel floor. Angle residuals wrap at +/-pi.
NIS uses the prior innovation and the eight-dimensional 95% threshold of
15.5073. The filter also supports exact fixed states without inverting a
singular covariance. UV diagnostics use `residual_left_*` and
`residual_right_*`, where `*` is `angle`, `u`, `v` or `length`. Position/length
residuals are pixels and angle residuals are radians.

Run the hardware-independent regression checks with:

```sh
cmake -S . -B build
cmake --build build --target uv_filter_test -j2
ctest --test-dir build -R '^uv_filter_test$' --output-on-failure
```
