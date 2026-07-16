{ pkgs, ... }:

{
  packages = with pkgs; [
    coreutils
    curl
    git
    gnugrep
    gnumake
    gnused
    jq
  ];
}
