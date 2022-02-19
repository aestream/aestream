{
  description = "Address Event Streaming library";

  inputs.nixpkgs.url = "nixpkgs/nixos-21.11";
  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils, mach-nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        py = pkgs.python39Packages;
        aestream = pkgs.stdenv.mkDerivation {
          name = "aestream";
          version = "0.1.0";
          src = ./.;
          nativeBuildInputs = [
            pkgs.cmake
            pkgs.gcc
          ];
          buildInputs = [
            pkgs.flatbuffers
            pkgs.ninja
            pkgs.lz4
          ];
          configurePhase = "cmake -GNinja -Bbuild/ .";
          buildPhase = "ninja -C build";
          installPhase = ''
            mkdir -p $out/lib $out/bin
            mv build/src/**/*.so $out/lib/
            mv build/src/*.so $out/lib/
            mv build/src/aestream $out/bin/
          '';
        };
        aestream-python = mach-nix.lib.${system}.buildPythonPackage {
          pname = "aestream";
          version = "0.1.0";
          src = ./.;
          requirements = ''
            numpy
            torch
          '';

          nativeBuildInputs = [
            pkgs.which
          ];
        };
      in
      rec {
        defaultPackage = aestream;
        packages.aestream-test = aestream-test;
        devShell = aestream-python;
      }
    );
}
