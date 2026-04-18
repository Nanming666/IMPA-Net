# Sample NetCDF Format

This package assumes a single NetCDF file such as `sample_cases.nc` containing `1-2` representative large-area convective cases.

## Recommended Dimensions

```text
case = 1 or 2
t_in = input sequence length
t_out = forecast sequence length
y = grid height
x = grid width
```

## Recommended Variables

Required or strongly recommended:

- `radar_in(case, t_in, y, x)`
- `radar_out(case, t_out, y, x)`
- `precip_in(case, t_in, y, x)`
- `topography(y, x)`
- `along_slope_wind(y, x)`

Optional but useful:

- `lat(y, x)`
- `lon(y, x)`
- `time_in(case, t_in)`
- `time_out(case, t_out)`
- `case_id(case)`

## Variable Semantics

- `radar_in`: past radar reflectivity sequence used as the main input
- `radar_out`: future radar reflectivity sequence used as the reference target
- `precip_in`: past precipitation field used as a dynamic auxiliary input
- `topography`: static terrain field
- `along_slope_wind`: static or case-level meteorological auxiliary field aligned to the terrain slope

## Recommended Numeric Type

Use `float32` for gridded variables:

- `radar_in`
- `radar_out`
- `precip_in`
- `topography`
- `along_slope_wind`
- `lat`
- `lon`

## Recommended Attributes

For each gridded variable, add attributes when possible:

- `long_name`
- `units`
- `normalization`
- `valid_range`
- `description`

Example:

```text
radar_in:
  long_name = "input radar reflectivity sequence"
  units = "normalized reflectivity"
  normalization = "value / 85.0"
```

## Shape Example

```text
radar_in(case, t_in, y, x) = (2, 20, 480, 560)
precip_in(case, t_in, y, x) = (2, 20, 480, 560)
radar_out(case, t_out, y, x) = (2, 20, 480, 560)
topography(y, x) = (480, 560)
along_slope_wind(y, x) = (480, 560)
```

## Why Static Fields Are Stored As 2D

`topography` and `along_slope_wind` are better stored as `[y, x]` instead of repeating them over time:

- smaller file size
- clearer semantics
- easier inspection by reviewers and editors
- the sample code expands them across time when needed
