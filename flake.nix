{
  description = "Address Event Streaming library";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-21.11";
    flake-utils.url = "github:numtide/flake-utils";
    mach-nix.url = "mach-nix/3.5.0";
  };

  outputs = { self, nixpkgs, flake-utils, mach-nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        py = pkgs.python39Packages;
        python-requirements = builtins.readFile ./requirements.txt;
        libcaer = pkgs.stdenv.mkDerivation {
          pname = "libcaer";
          version = "1.0";
          src = pkgs.fetchFromGitLab {
            owner = "dv";
            group = "inivation";
            repo = "libcaer";
            rev = "3.3.14";
            hash = "sha1-ZszisfBWVLM7cXCGq7y1FeJ3RJA=";
          };
          nativeBuildInputs = with pkgs; [
            pkg-config libusb1 cmake gcc flatbuffers ninja lz4
          ];
        };
        libtorch = pkgs.stdenv.mkDerivation {
          pname = "libtorch";
          version = "1.12.0";
          src = pkgs.fetchzip {
            name = "libtorch";
            url = "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.12.0%2Bcpu.zip";
            hash = "sha256-coeCeX8OYQyMhLjDurZjXs1uII25xaOC51XmOK3uMTk=";
          };
          nativeBuildInputs = [ pkgs.unzip ];
          buildPhase = ''
            mkdir $out
            mv * $out/
          '';
          phases = [ "unpackPhase" "buildPhase" ];
        };
        aestream = pkgs.stdenv.mkDerivation {
          name = "aestream";
          version = "0.3.0";
          src = ./.;
          nativeBuildInputs = [
            pkgs.cmake
            pkgs.gcc
            pkgs.autoPatchelfHook
          ];
          buildInputs = [
            pkgs.flatbuffers
            pkgs.python39
            pkgs.ninja
            pkgs.lz4
            libcaer
            libtorch
          ];
          cmakeFlags = [
            "-GNinja"
            "-DCMAKE_PREFIX_PATH=${libtorch}"
            "-DFLATBUFFERS_SOURCE_DIR=${pkgs.flatbuffers.src}"
          ];
          preBuild = ''
            addAutoPatchelfSearchPath src/
            addAutoPatchelfSearchPath src/input
            addAutoPatchelfSearchPath src/output
          '';
          installPhase = ''
            install -m555 -D -t $out/lib/ src/*.so src/input/*.so src/output/*.so
            install -m755 -D src/aestream $out/bin/aestream
          '';
        };
        aestream-test = aestream.overrideAttrs (parent: {
          name = "aestream_test";
          nativeBuildInputs = parent.nativeBuildInputs ++ [
            pkgs.makeWrapper
          ];
          buildInputs = parent.buildInputs ++ [
            pkgs.gtest
          ];
          cmakeFlags = parent.cmakeFlags ++ [
            "-DCMAKE_BUILD_TYPE=Debug"
            "-DCMAKE_PREFIX_PATH=${libtorch};${pkgs.gtest}"
          ];
          installPhase = parent.installPhase + ''
            install -m555 -D $src/example/davis.aedat4 $out/example/davis.aedat4
            install -m755 -D test/aestream_test $out/bin/aestream_test
          '';
        });
        aestream-python = mach-nix.lib.${system}.buildPythonPackage {
          pname = "aestream";
          version = "0.3.0";
          src = ./.;
          requirements = python-requirements;

          buildInputs = [ libcaer py.pytorch ];
          nativeBuildInputs = [
            pkgs.which
          ];
        };
      in
      rec {
        devShells = flake-utils.lib.flattenTree {
          default = aestream-python;
        };
        packages = flake-utils.lib.flattenTree {
          default = aestream;
          test = aestream-test;
        };
      }
    );
}
