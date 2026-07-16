{ lib, ... }:

{
  # NVIDIA's host driver provides libcuda.so. Keep toolkit libraries out of this
  # runtime profile so pip/uv CUDA wheels keep using their matching libraries.
  env.LD_LIBRARY_PATH = lib.mkDefault "/run/opengl-driver/lib";

  scripts.gpu-run = {
    description = "Run a command on the NVIDIA GPU through PRIME offload";
    exec = ''
      exec nvidia-offload "$@"
    '';
  };
}
