#!/usr/bin/env python3

import os
import sys
import time
import shutil
import random
import subprocess
import urllib.request
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

PROJECT_DIR = Path.cwd()
RESULT_DIR = PROJECT_DIR / "experiment_results"
IMAGE_DIR = PROJECT_DIR / "experiment_images"

CANNY_EXE = PROJECT_DIR / "canny"

# Image sizes required for your project
IMAGES = [
    ("image_720p", 1280, 720),
    ("image_1080p", 1920, 1080),
    ("image_4k", 3840, 2160),
]

# These URLs should return images at the exact requested dimensions.
# If downloading fails, the script generates exact-size synthetic images.
DOWNLOAD_URLS = {
    "image_720p": "https://picsum.photos/seed/cuda_canny_720/1280/720",
    "image_1080p": "https://picsum.photos/seed/cuda_canny_1080/1920/1080",
    "image_4k": "https://picsum.photos/seed/cuda_canny_4k/3840/2160",
}

# Main modes to compare
MAIN_EXPERIMENTS = [
    ("cpu",    "--mode cpu"),
    ("dense",  "--mode dense --compare-cpu"),
    ("opt",    "--mode opt --compare-cpu"),
    ("sparse", "--mode sparse --compare-cpu"),
]

# Sparse threshold experiments to test edge density effect
SPARSE_THRESHOLD_EXPERIMENTS = [
    ("sparse_low_threshold",  "--mode sparse --low 20 --high 60 --compare-cpu"),
    ("sparse_mid_threshold",  "--mode sparse --low 50 --high 100 --compare-cpu"),
    ("sparse_high_threshold", "--mode sparse --low 80 --high 160 --compare-cpu"),
]

# Use 10 for final results. Lower this only if testing.
REPEAT = 10


# ============================================================
# Utility functions
# ============================================================

def run_command(cmd, log_file, cwd=None, allow_fail=False):
    """
    Run shell command, record stdout/stderr into log file.
    """
    if cwd is None:
        cwd = PROJECT_DIR

    log_file.write("\n" + "=" * 90 + "\n")
    log_file.write(f"COMMAND: {cmd}\n")
    log_file.write("=" * 90 + "\n")
    log_file.flush()

    print(f"\n[Running] {cmd}")

    start = time.time()

    proc = subprocess.run(
        cmd,
        shell=True,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )

    elapsed = time.time() - start

    log_file.write(proc.stdout)
    log_file.write(f"\n[Command exit code] {proc.returncode}\n")
    log_file.write(f"[Wall time] {elapsed:.3f} seconds\n")
    log_file.flush()

    print(proc.stdout)
    print(f"[Finished] exit={proc.returncode}, wall_time={elapsed:.3f}s")

    if proc.returncode != 0 and not allow_fail:
        raise RuntimeError(f"Command failed: {cmd}")

    return proc.returncode, proc.stdout


def have_command(name):
    return shutil.which(name) is not None


def ensure_dirs():
    RESULT_DIR.mkdir(exist_ok=True)
    IMAGE_DIR.mkdir(exist_ok=True)


def download_file(url, output_path, timeout=60):
    """
    Download file using only Python standard library.
    """
    print(f"[Download] {url}")
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 CUDA-Canny-Experiment"
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = response.read()

    with open(output_path, "wb") as f:
        f.write(data)

    print(f"[Saved] {output_path} ({len(data)} bytes)")


def convert_to_pgm(input_image, output_pgm):
    """
    Convert downloaded jpg/png image to grayscale PGM.
    Tries ImageMagick first.
    """
    if have_command("magick"):
        cmd = f'magick "{input_image}" -resize "!{output_pgm.stem.split("_")[-2] if False else ""}" -colorspace Gray "{output_pgm}"'
        # Use simpler command because downloaded image should already be exact size.
        cmd = f'magick "{input_image}" -colorspace Gray "{output_pgm}"'
        subprocess.check_call(cmd, shell=True)
        return True

    if have_command("convert"):
        cmd = f'convert "{input_image}" -colorspace Gray "{output_pgm}"'
        subprocess.check_call(cmd, shell=True)
        return True

    return False


def generate_synthetic_pgm(output_pgm, width, height):
    """
    Generate exact-size PGM image with many edges.
    No external packages needed.
    """
    print(f"[Fallback] Generating synthetic image: {output_pgm}")

    random.seed(width * 1000 + height)

    with open(output_pgm, "wb") as f:
        f.write(f"P5\n{width} {height}\n255\n".encode())

        for y in range(height):
            row = bytearray(width)

            for x in range(width):
                # Smooth background gradient
                v = int((x / max(1, width - 1)) * 180 + (y / max(1, height - 1)) * 50)

                # Large rectangle
                if width * 0.18 < x < width * 0.82 and height * 0.20 < y < height * 0.75:
                    v = 210

                # Rectangle border
                border = 5
                if (
                    abs(x - int(width * 0.18)) < border or
                    abs(x - int(width * 0.82)) < border
                ) and height * 0.20 < y < height * 0.75:
                    v = 20

                if (
                    abs(y - int(height * 0.20)) < border or
                    abs(y - int(height * 0.75)) < border
                ) and width * 0.18 < x < width * 0.82:
                    v = 20

                # Circle-like edge
                cx, cy = width * 0.50, height * 0.48
                r = min(width, height) * 0.18
                dist2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
                if abs(dist2 - r * r) < r * 18:
                    v = 255

                # Diagonal bright line
                if abs(y - (height * 0.15 + x * height / width * 0.7)) < 3:
                    v = 255

                # Vertical stripe edges
                if x % max(40, width // 30) in (0, 1):
                    if height * 0.05 < y < height * 0.95:
                        v = 40

                row[x] = max(0, min(255, int(v)))

            f.write(row)


def prepare_images(log_file):
    """
    Download exact-size images and convert to PGM.
    If this fails, generate synthetic PGM images.
    """
    log_file.write("\n\n")
    log_file.write("# Image Preparation\n")
    log_file.write("=" * 90 + "\n")

    prepared = []

    for name, width, height in IMAGES:
        jpg_path = IMAGE_DIR / f"{name}_{width}x{height}.jpg"
        pgm_path = IMAGE_DIR / f"{name}_{width}x{height}.pgm"

        if pgm_path.exists():
            print(f"[Image exists] {pgm_path}")
            log_file.write(f"Using existing PGM: {pgm_path}\n")
            prepared.append((name, width, height, pgm_path))
            continue

        downloaded = False
        converted = False

        try:
            url = DOWNLOAD_URLS[name]
            download_file(url, jpg_path)
            downloaded = True
            log_file.write(f"Downloaded {name} from {url}\n")
        except Exception as e:
            log_file.write(f"Download failed for {name}: {e}\n")
            print(f"[Warning] Download failed for {name}: {e}")

        if downloaded:
            try:
                converted = convert_to_pgm(jpg_path, pgm_path)
                if converted:
                    log_file.write(f"Converted to PGM: {pgm_path}\n")
                else:
                    log_file.write("ImageMagick not found, cannot convert downloaded image.\n")
            except Exception as e:
                log_file.write(f"Conversion failed for {name}: {e}\n")
                print(f"[Warning] Conversion failed for {name}: {e}")

        if not pgm_path.exists():
            generate_synthetic_pgm(pgm_path, width, height)
            log_file.write(f"Generated fallback synthetic PGM: {pgm_path}\n")

        prepared.append((name, width, height, pgm_path))

    log_file.flush()
    return prepared


def compile_project(log_file):
    """
    Compile CUDA project.
    Tries default Makefile first, then common CUDA architectures.
    """
    log_file.write("\n\n")
    log_file.write("# Compilation\n")
    log_file.write("=" * 90 + "\n")
    log_file.flush()

    if not have_command("make"):
        raise RuntimeError("make command not found. Cannot compile.")

    if not have_command("nvcc"):
        raise RuntimeError("nvcc command not found. CUDA compiler is not available in PATH.")

    # Clean first
    run_command("make clean", log_file, allow_fail=True)

    # Try default make
    code, _ = run_command("make", log_file, allow_fail=True)
    if code == 0 and CANNY_EXE.exists():
        return

    # Try common architectures
    arch_list = ["sm_90", "sm_89", "sm_86", "sm_80", "sm_75", "sm_70", "sm_61", "sm_60"]

    for arch in arch_list:
        run_command("make clean", log_file, allow_fail=True)
        code, _ = run_command(f"make ARCH={arch}", log_file, allow_fail=True)
        if code == 0 and CANNY_EXE.exists():
            log_file.write(f"\nCompilation succeeded with ARCH={arch}\n")
            log_file.flush()
            return

    raise RuntimeError("Compilation failed for default and common CUDA architectures.")


def run_experiments(prepared_images, log_file):
    """
    Run all CPU/GPU experiments and save output files.
    """
    log_file.write("\n\n")
    log_file.write("# Experiments\n")
    log_file.write("=" * 90 + "\n")
    log_file.flush()

    if not CANNY_EXE.exists():
        raise RuntimeError("./canny executable not found. Compile failed or wrong directory.")

    summary_csv = RESULT_DIR / "summary_commands.csv"

    with open(summary_csv, "w") as csv:
        csv.write("image,width,height,experiment,input,output,command\n")

        for name, width, height, pgm_path in prepared_images:
            image_result_dir = RESULT_DIR / name
            image_result_dir.mkdir(exist_ok=True)

            log_file.write("\n\n")
            log_file.write("#" * 90 + "\n")
            log_file.write(f"# Image: {name}, size={width}x{height}\n")
            log_file.write("#" * 90 + "\n")
            log_file.flush()

            # Main experiments
            for exp_name, args in MAIN_EXPERIMENTS:
                output_pgm = image_result_dir / f"{name}_{exp_name}.pgm"

                cmd = f'"{CANNY_EXE}" "{pgm_path}" "{output_pgm}" {args} --repeat {REPEAT}'

                csv.write(
                    f'{name},{width},{height},{exp_name},"{pgm_path}","{output_pgm}","{cmd}"\n'
                )
                csv.flush()

                run_command(cmd, log_file, allow_fail=True)

            # Sparse threshold experiments
            for exp_name, args in SPARSE_THRESHOLD_EXPERIMENTS:
                output_pgm = image_result_dir / f"{name}_{exp_name}.pgm"

                cmd = f'"{CANNY_EXE}" "{pgm_path}" "{output_pgm}" {args} --repeat {REPEAT}'

                csv.write(
                    f'{name},{width},{height},{exp_name},"{pgm_path}","{output_pgm}","{cmd}"\n'
                )
                csv.flush()

                run_command(cmd, log_file, allow_fail=True)

    log_file.write(f"\nSummary command CSV saved to: {summary_csv}\n")
    log_file.flush()


def write_readme():
    readme = RESULT_DIR / "README_RESULTS.txt"
    with open(readme, "w") as f:
        f.write(
            """CUDA Canny Experiment Results

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
"""
        )


def main():
    ensure_dirs()

    log_path = RESULT_DIR / "experiment_log.txt"

    with open(log_path, "w") as log_file:
        log_file.write("CUDA Canny Full Experiment Log\n")
        log_file.write("=" * 90 + "\n")
        log_file.write(f"Project directory: {PROJECT_DIR}\n")
        log_file.write(f"Result directory: {RESULT_DIR}\n")
        log_file.write(f"Image directory: {IMAGE_DIR}\n")
        log_file.write(f"Repeat count: {REPEAT}\n")
        log_file.write(f"Start time: {time.ctime()}\n")
        log_file.flush()

        # Record system info
        log_file.write("\n\n# System Info\n")
        log_file.write("=" * 90 + "\n")
        log_file.flush()

        run_command("hostname", log_file, allow_fail=True)
        run_command("date", log_file, allow_fail=True)
        run_command("which nvcc && nvcc --version", log_file, allow_fail=True)
        run_command("which nvidia-smi && nvidia-smi", log_file, allow_fail=True)

        # Prepare images
        prepared_images = prepare_images(log_file)

        # Compile
        compile_project(log_file)

        # Run all experiments
        run_experiments(prepared_images, log_file)

        log_file.write("\n\n")
        log_file.write("=" * 90 + "\n")
        log_file.write(f"All experiments finished at: {time.ctime()}\n")
        log_file.write(f"Main log saved to: {log_path}\n")
        log_file.write("=" * 90 + "\n")
        log_file.flush()

    write_readme()

    print("\n" + "=" * 90)
    print("All experiments finished.")
    print(f"Main log: {log_path}")
    print(f"Results folder: {RESULT_DIR}")
    print("=" * 90)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERROR]")
        print(e)
        print("\nCheck experiment_results/experiment_log.txt if it was created.")
        sys.exit(1)
