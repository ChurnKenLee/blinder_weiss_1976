{ pkgs, ... }:

{
  languages.python.enable = true;

  packages = with pkgs; [
    uv
    gcc
    pkg-config
    patchelf
  ];

  env = {
    UV_MANAGED_PYTHON = "true";
    UV_PROJECT_ENVIRONMENT = ".venv";
  };
}
