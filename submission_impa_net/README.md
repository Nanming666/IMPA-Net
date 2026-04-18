# IMPA-Net Sample Repository

This repository provides a compact sample implementation and a representative large-area example dataset accompanying the manuscript:

**IMPA-Net: Meteorology-Aware Multi-Scale Attention and Dynamic Loss for Extreme Convective Radar Nowcasting**

The material in this repository is intended for editorial and reviewer inspection during manuscript submission. It is designed to present the central method components and the expected data organization in a concise and inspectable form.

## Repository Purpose

This is a **sample submission package**, rather than the full internal research codebase.

The repository is intended to provide:

- a compact, self-contained example of the core model design
- a representative NetCDF sample for one large-area convective case
- a clear description of the data variables and repository scope

Accordingly, the package preserves the method-level ideas most relevant to the manuscript while omitting the broader engineering framework used for large-scale training and experimentation.

## Included Files

- `sample_method.py`: a self-contained PyTorch example illustrating the main method components, including meteorology-aware multi-channel fusion, multi-scale spatial encoding, temporal sequence mixing, and a dynamic extreme-aware loss.
- `sample_cases.nc`: a representative large-area sample case prepared for repository inspection.
- `DATA_FORMAT.md`: documentation of the recommended NetCDF variable layout and metadata conventions.
- `PACKAGE_SCOPE.md`: a short note clarifying what is intentionally preserved in this repository and what is intentionally omitted.
- `UPLOAD_CHECKLIST.md`: a practical checklist for final repository review before publication or sharing.

## Method Summary

The compact sample implementation reflects the main ideas emphasized in the manuscript:

- meteorology-aware fusion of radar and auxiliary meteorological inputs
- multi-scale spatial feature extraction for convective structure representation
- temporal mixing for sequence-to-sequence nowcasting
- dynamic loss weighting with explicit emphasis on extreme convective echoes

## Sample Data

The current sample NetCDF file contains one representative example case.

- `case_id = 1715`
- input sequence length: `20` frames
- forecast reference length: `20` frames
- spatial coverage: full large-area field at `480 x 560`

The sample file includes:

- `radar_in`
- `precip_in`
- `radar_out`
- `topography`
- `along_slope_wind`
- `lat`
- `lon`
- case and frame traceability metadata

## Minimal Requirements

The sample script only requires a small set of dependencies:

- Python 3.9+
- PyTorch
- NumPy
- xarray
- a NetCDF backend supported by xarray, such as `netcdf4`

Example installation:

```bash
pip install torch numpy xarray netcdf4
```

## Example Usage

```bash
python sample_method.py --nc_path sample_cases.nc --case_index 0 --device cpu
```

The script reads the NetCDF sample, constructs the compact IMPA-Net example, performs a forward pass, and computes a sample dynamic loss when `radar_out` is available.

## Scope Statement

This repository should be interpreted as a concise method-and-data companion for manuscript submission. It is not intended to serve as a complete training, benchmarking, or reproduction framework.

Readers seeking the exact repository scope should refer to `PACKAGE_SCOPE.md`, and readers preparing additional NetCDF examples should refer to `DATA_FORMAT.md`.
