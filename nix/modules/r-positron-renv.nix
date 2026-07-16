{ lib, pkgs, ... }:

let
  renvBootstrap = pkgs.rPackages.renv;
  # System libraries commonly required when CRAN packages are compiled from
  # source (for example, tidyverse, ragg, and xml2). These are not R packages.
  rNativeDeps = with pkgs; [
    cmake
    curl
    fontconfig
    freetype
    fribidi
    harfbuzz
    libjpeg
    libpng
    libtiff
    libuv
    libwebp
    libxml2
    openssl
    gdal
    geos
    proj
    sqlite
    udunits
    zlib
  ];
in
{
  # Positron needs the plain R launcher on PATH. All R packages, including
  # renv itself after initialization, belong to the project-local renv library.
  # Mixing rPackages into R_LIBS_SITE defeats renv's isolation and can load a
  # package compiled for a different R ABI.
  packages = [
    pkgs.R
    pkgs.gfortran
  ]
  ++ rNativeDeps;

  # pkg-config finds headers at build time; the loader needs the same libraries
  # at install and runtime. Retain the NVIDIA driver path for GPU projects.
  env = {
    PKG_CONFIG_PATH = lib.makeSearchPathOutput "dev" "lib/pkgconfig" rNativeDeps;
    LD_LIBRARY_PATH = "${lib.makeLibraryPath rNativeDeps}:/run/opengl-driver/lib";
    # Nix's udunits package does not ship a pkg-config file. The CRAN `units`
    # configure script explicitly supports these two variables.
    UDUNITS2_INCLUDE = "${pkgs.udunits}/include";
    UDUNITS2_LIBS = "-L${pkgs.udunits}/lib -ludunits2";
  };

  scripts.r-renv-init = {
    description = "Bootstrap an isolated renv project in the current directory";
    exec = ''
      R --vanilla --quiet -e 'library(renv, lib.loc = "${renvBootstrap}/library"); renv::init(bare = TRUE)'
    '';
  };

  scripts.r-renv-restore = {
    description = "Restore R packages from renv.lock";
    exec = ''
      R --quiet -e 'renv::restore(prompt = FALSE)'
    '';
  };
}
