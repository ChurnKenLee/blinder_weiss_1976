{ pkgs, ... }:

let
  cuda = pkgs.cudaPackages;

  cudaRoot = pkgs.buildEnv {
    name = "cuda-project-root";
    paths = with cuda; [
      cuda_nvcc
      cuda_cudart
      cuda_nvrtc
      cuda_nvvm
      cccl
    ];
    pathsToLink = [
      "/bin"
      "/include"
      "/lib"
      "/lib64"
      "/nvvm"
      "/share"
    ];
    extraOutputsToInstall = [
      "out"
      "dev"
      "lib"
      "stubs"
    ];
    ignoreCollisions = true;
  };

  cudaDoctor = pkgs.writeShellApplication {
    name = "cuda-doctor";
    runtimeInputs = with pkgs; [
      coreutils
      gcc
      gnugrep
    ];
    text = ''
      set -euo pipefail

      echo "CUDA_HOME=$CUDA_HOME"
      command -v nvcc
      "$NVCC" --version | head -n 4
      test -f "$CUDA_HOME/include/cuda_runtime.h"
      test -e "$CUDA_HOME/nvvm/libdevice/libdevice.10.bc"
      test -e /run/opengl-driver/lib/libcuda.so

      tmp="$(mktemp -d)"
      trap 'rm -rf "$tmp"' EXIT
      printf '#include <cuda_runtime.h>\nint main() { return 0; }\n' > "$tmp/check.cu"
      "$NVCC" -c "$tmp/check.cu" -o "$tmp/check.o"
      echo "cuda-doctor: ok"
    '';
  };

  cudaLinkLibraryPath = "${cudaRoot}/lib:${cudaRoot}/lib64:${cudaRoot}/lib/stubs:${cudaRoot}/lib64/stubs";
in
{
  packages = [
    cudaRoot
    cudaDoctor
  ];

  env = {
    CUDA_HOME = "${cudaRoot}";
    CUDA_PATH = "${cudaRoot}";
    CUDA_ROOT = "${cudaRoot}";
    CUDAToolkit_ROOT = "${cudaRoot}";
    CUDACXX = "${cudaRoot}/bin/nvcc";
    NVCC = "${cudaRoot}/bin/nvcc";
    CPATH = "${cudaRoot}/include";
    LIBRARY_PATH = "/run/opengl-driver/lib:${cudaLinkLibraryPath}";
    CMAKE_PREFIX_PATH = "${cudaRoot}";
    CMAKE_LIBRARY_PATH = "/run/opengl-driver/lib:${cudaLinkLibraryPath}";
    EXTRA_CCFLAGS = "-I${cudaRoot}/include";
    EXTRA_LDFLAGS = "-L/run/opengl-driver/lib -L${cudaRoot}/lib -L${cudaRoot}/lib64 -L${cudaRoot}/lib/stubs";
    TRITON_PTXAS_PATH = "${cudaRoot}/bin/ptxas";
    TRITON_PTXAS_BLACKWELL_PATH = "${cudaRoot}/bin/ptxas";
  };
}
