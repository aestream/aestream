{
  description = "Address Event Streaming library";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-22.11";
    flake-utils.url = "github:numtide/flake-utils";
    mach-nix.url = "mach-nix/3.5.0";
    # mach-nix.pypiDataRev = "322a4f20c357704644abe8c2e50412e9b9c16909";
  };

  outputs = { self, nixpkgs, flake-utils, mach-nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        py = pkgs.python310Packages;
        libcaer = pkgs.stdenv.mkDerivation {
          pname = "libcaer";
          version = "3.3.14";
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
          preBuild = ''
            substituteInPlace libcaer.pc --replace // /
          '';
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
            py.torch
            libcaer
          ];
          cmakeFlags = [
            "-GNinja"
            "-DCMAKE_PREFIX_PATH=${py.torch}"
            "-DCMAKE_SKIP_BUILD_RPATH=ON"
            "-DFLATBUFFERS_SOURCE_DIR=${pkgs.flatbuffers.src}"
          ];
          preBuild = ''
            addAutoPatchelfSearchPath src/
            addAutoPatchelfSearchPath src/file
            addAutoPatchelfSearchPath src/input
            addAutoPatchelfSearchPath src/output
          '';
          installPhase = ''
            install -m555 -D -t $out/lib/ src/*.so src/file/*.so src/input/*.so src/output/*.so
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
            "-DCMAKE_PREFIX_PATH=${py.torch};${pkgs.gtest}"
          ];
          installPhase = parent.installPhase + ''
            install -m555 -D $src/example/sample.aedat4 $out/example/sample.aedat4
            install -m555 -D $src/example/sample.dat $out/example/sample.dat
            install -m755 -D test/aestream_test $out/bin/aestream_test
          '';
        });
        aestream-python = mach-nix.lib.${system}.buildPythonPackage {
          pname = "aestream";
          version = "0.4.0";
          src = ./.;
          requirements = "scikit-build\nnumpy";
          providers.pip = "wheel";

          buildInputs = [ pkgs.lz4 pkgs.zlib py.pybind11 libcaer pkgs.torch ];
          nativeBuildInputs = [
            pkgs.cmake 
            pkgs.which
          ];
          python = "python310";
          # pypiDataRev = "c8a55398a0e24b5560732cc94cb24172eaddf72f";
          # pypiDataSha256 = "sha256-8geJawMHqrwk/+Dvx5pkm/T9BzVJPFqN0leHe3VSsQg=";
          postShellHook = ''
              echo ${pkgs.torch}
          #   export LD_LIBRARY_PATH=${pkgs.glibc.bin}:$LD_LIBRARY_PATH
          '';
        };
      in
      rec {
        devShells = flake-utils.lib.flattenTree {
          default = aestream;
          python = aestream-python;
        };
        packages = flake-utils.lib.flattenTree {
          default = aestream;
          test = aestream-test;
        };
      }
    );
}
