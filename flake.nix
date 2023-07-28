{
  description = "Address Event Streaming library";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-23.05";
    flake-utils.url = "github:numtide/flake-utils";
    # mach-nix.url = "mach-nix/3.5.0";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        py = pkgs.python310Packages { inherit pkgs; };
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
            pkg-config
            libusb1
            cmake
            gcc
            flatbuffers
            ninja
            lz4
          ];
          preBuild = ''
            substituteInPlace libcaer.pc --replace // /
          '';
        };
        aestream = pkgs.stdenv.mkDerivation {
          name = "aestream";
          version = "0.6.1";
          src = ./.;
          nativeBuildInputs = [
            pkgs.pkg-config
            pkgs.cmake
            pkgs.gcc
            pkgs.autoPatchelfHook
          ];
          buildInputs = [
            pkgs.libusb1
            pkgs.flatbuffers
            pkgs.python39
            pkgs.ninja
            pkgs.lz4
            pkgs.SDL2
            pkgs.zeromq
            pkgs.cppzmq
            libcaer
          ];
          cmakeFlags = [
            "-GNinja"
            "-DCMAKE_SKIP_BUILD_RPATH=ON"
            "-DFLATBUFFERS_SOURCE_DIR=${pkgs.flatbuffers.src}"
            # "-DCMAKE_PREFIX_PATH=${openeb.src}/sdk/"
          ];

          preBuild = ''
            addAutoPatchelfSearchPath src/
            addAutoPatchelfSearchPath src/file
            addAutoPatchelfSearchPath src/input
            addAutoPatchelfSearchPath src/output
          '';
          installPhase = ''
            install -m555 -D -t $out/lib/ src/cpp/*.a src/cpp/file/*.a src/cpp/input/*.a src/cpp/output/*.a
            install -m755 -D src/cpp/aestream $out/bin/aestream
          '';
        };
        aestream-test = aestream.overrideAttrs (parent: {
          name = "aestream_test";
          nativeBuildInputs = parent.nativeBuildInputs ++ [
            pkgs.makeWrapper
          ];
          buildInputs = parent.buildInputs ++ [
            pkgs.gdb
            pkgs.gtest
          ];
          cmakeFlags = parent.cmakeFlags ++ [
            "-DCMAKE_BUILD_TYPE=Debug"
            "-DCMAKE_PREFIX_PATH=${pkgs.gtest}"
          ];
          installPhase = parent.installPhase + ''
            install -m555 -D $src/example/sample.aedat4 $out/example/sample.aedat4
            install -m555 -D $src/example/sample.dat $out/example/sample.dat
            install -m755 -D test/aestream_test $out/bin/aestream_test
          '';
        });
        aestream-python =
          let
            pypkgs = pkgs.python3Packages;
          in
          pkgs.mkShell {
            buildInputs = [
              pkgs.lz4
              pkgs.libsodium
              pkgs.zlib
              pkgs.cmake
              pkgs.ninja
              pkgs.cpm-cmake
              pkgs.flatcc
              pkgs.zeromq
              pkgs.cppzmq
              libcaer
              pypkgs.python
              pypkgs.pip
              pypkgs.venvShellHook
              pypkgs.numpy
              pypkgs.pysdl2
              pypkgs.scipy
            ];
            venvDir = "./.venv";
            postVenvCreation = ''
              pip install scikit-build-core nanobind
            '';
            postShellHook = ''
              unset SOURCE_DATE_EPOCH
            '';
          };
      in
      rec {
        devShells = flake-utils.lib.flattenTree {
          default = aestream.overrideAttrs (parent: {
            buildInputs = parent.buildInputs ++ [ pkgs.clang-tools ];
          });
          test = aestream-test;
          python = aestream-python;
        };
        packages = flake-utils.lib.flattenTree {
          default = aestream;
          test = aestream-test;
        };
      }
    );
}
