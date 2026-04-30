# CUDA Canny Edge Detection Project

This project includes a CPU baseline, a dense CUDA pipeline, an optimized CUDA pipeline, and a sparse CUDA pipeline with stream compaction.

## Build

```bash
make
```

If the default GPU architecture does not work, run:

```bash
make clean
make ARCH=sm_75
```

Common architecture choices: `sm_70`, `sm_75`, `sm_80`, `sm_86`, `sm_89`.

## Convert an image to PGM

```bash
python3 convert_to_pgm.py input.jpg input.pgm
```

## Run

```bash
./canny input.pgm out_cpu.pgm --mode cpu
./canny input.pgm out_dense.pgm --mode dense --compare-cpu --repeat 10
./canny input.pgm out_opt.pgm --mode opt --compare-cpu --repeat 10
./canny input.pgm out_sparse.pgm --mode sparse --compare-cpu --repeat 10
```

## Test edge density

Change thresholds:

```bash
./canny input.pgm out_sparse_low.pgm --mode sparse --low 20 --high 60 --repeat 10 --compare-cpu
./canny input.pgm out_sparse_mid.pgm --mode sparse --low 50 --high 100 --repeat 10 --compare-cpu
./canny input.pgm out_sparse_high.pgm --mode sparse --low 80 --high 160 --repeat 10 --compare-cpu
```

Lower thresholds usually create more candidate pixels. Higher thresholds usually create fewer candidate pixels.

## Modes

- `cpu`: CPU reference implementation.
- `dense`: dense CUDA pipeline using a direct 5x5 Gaussian blur.
- `opt`: optimized CUDA pipeline using separable Gaussian blur and constant memory for masks.
- `sparse`: optimized CUDA pipeline plus threshold mask, exclusive scan, scatter, and sparse hysteresis.
