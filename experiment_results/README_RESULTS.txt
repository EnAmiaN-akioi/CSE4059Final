CUDA Canny Experiment Results

Important files:
1. experiment_log.txt
   Full terminal output from compilation and all experiments.

2. summary_commands.csv
   List of all commands that were executed.

3. experiment_images/
   Input images in PGM format.

4. experiment_results/image_720p/
   Output edge maps for 720p.

5. experiment_results/image_1080p/
   Output edge maps for 1080p.

6. experiment_results/image_4k/
   Output edge maps for 4K.

For your report, use experiment_log.txt to extract:
- CPU runtime
- Dense CUDA runtime
- Optimized CUDA runtime
- Sparse CUDA runtime
- Per-stage timing
- Candidate pixel count
- CPU/GPU correctness comparison
- Speedup vs CPU

Recommended discussion:
- GPU speedup should become more meaningful as image size increases.
- Optimized CUDA should improve stencil-heavy stages such as Gaussian and Sobel.
- Sparse compaction helps only when candidate edge density is low enough.
- If many pixels survive thresholding, scan and scatter overhead can make sparse slower.
