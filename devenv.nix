{ lib, ... }:

let
  project = import ./nix/project.nix;
in
{
  # Secrets stay in direnv's process environment; do not enable devenv.dotenv,
  # which copies the .env file into the Nix store.
  dotenv.disableHint = true;

  imports = [
    ./nix/modules/base.nix
  ]
  ++ lib.optionals project.python [ ./nix/modules/python-uv.nix ]
  ++ lib.optionals project.gpu [ ./nix/modules/gpu-runtime.nix ]
  ++ lib.optionals project.cudaDev [ ./nix/modules/cuda-dev.nix ]
  ++ lib.optionals (project.r == "renv") [ ./nix/modules/r-positron-renv.nix ]
  ++ lib.optionals (project.r == "nix") [ ./nix/modules/r-positron-nix.nix ];
}
