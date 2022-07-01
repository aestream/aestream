{
  description = "Address Event Streaming library";

  inputs.nixpkgs.url = "nixpkgs/nixos-21.11";
  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils, mach-nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        py = pkgs.python39Packages;
        python-requirements = builtins.readFile ./requirements.txt;
        libcaer = pkgs.stdenv.mkDerivation {
          pname = "libcaer";
          version = "1.0";
          src = pkgs.fetchurl {
            url = https://gitlab.com/inivation/dv/libcaer/-/archive/54191a3b27db4645ae3f83d96cf9bad5e1d646da/libcaer-54191a3b27db4645ae3f83d96cf9bad5e1d646da.tar.bz2;
            sha1 = "3b531ddda80513e169227b2a77823ad64f0aedfb";
          };
          nativeBuildInputs = with pkgs; [
            pkg-config libusb1 cmake gcc flatbuffers ninja lz4
          ];
        };
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
            pkgs.libtorch-bin
            libcaer
          ];
          configurePhase = "cmake -GNinja -Bbuild/ .";
          buildPhase = ''
            flatc --cpp -o src/ src/flatbuffers/*
            ninja -C build
            '';
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
          requirements = python-requirements;

          buildInputs = [ libcaer py.pytorch ];
          nativeBuildInputs = [
            pkgs.which
          ];
        };
      in
      rec {
        defaultPackage = aestream;
        devShell = aestream-python;
      }
    );
}
