{ pkgs, ... }:

{
  # Keep R separate from rPackages: Positron is incompatible with rWrapper.
  packages =
    with pkgs;
    [ R ]
    ++ (with rPackages; [
      dotenv
      languageserver
      here
      arrow
      haven
      readxl
      writexl
      foreign
      janitor
      tidyverse
      tidylog
      ggplot2
      collapse
      fixest
    ]);
}
