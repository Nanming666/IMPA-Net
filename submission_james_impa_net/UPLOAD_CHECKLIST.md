# Upload Checklist

Before pushing this folder to GitHub, check the following:

- Confirm the repository name and visibility
- Confirm that the manuscript title in `README.md` matches the current submission title
- Confirm that `sample_cases.nc` contains only `1-2` representative example cases
- Confirm that no full training dataset or institution-restricted data is included
- Confirm that the NetCDF variable names match `DATA_FORMAT.md`
- Confirm that `sample_method.py` runs on the shared sample file
- Confirm that any absolute local paths have been removed
- Confirm that reviewer-facing text does not promise full reproducibility if this package is only a sample release

Recommended final repository contents:

- `sample_method.py`
- `sample_cases.nc`
- `README.md`
- `DATA_FORMAT.md`
- `PACKAGE_SCOPE.md`
