{
  description = "Address Event Streaming library";

  inputs.nixpkgs.url = "nixpkgs/nixos-21.11";
  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.simpleFlake {
      inherit self nixpkgs;
      name = "aestream";
      shell = { pkgs ? import <nixpkgs> }:
        let py = pkgs.python39Packages;
        in
        pkgs.mkShell {
          buildInputs = [
            pkgs.cmake
            pkgs.flatbuffers
            pkgs.gcc
            pkgs.lz4
            pkgs.ninja
            py.pytorch
            py.pip
            py.pybind11
            # py.venvShellHook
          ];
          shellHook = ''
            export PIP_PREFIX=$(pwd)/_build/pip_packages
            export PYTHONPATH="$PIP_PREFIX/${pkgs.python3.sitePackages}:$PYTHONPATH"
            export PATH="$PIP_PREFIX/bin:$PATH"
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [pkgs.stdenv.cc.cc]}
            unset SOURCE_DATE_EPOCH
          '';
          # venvDir = "./.venv";
          # postShellHook = ''
          #   export LD_LIBRARY_PATH=${stdenv.cc.cc.lib}/lib/:/run/opengl-driver/lib/
          #   # allow pip to install wheels
          #   unset SOURCE_DATE_EPOCH
          # '';
        };
    };
}
