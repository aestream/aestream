{
  description = "Address Event Streaming library";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-22.11";
    flake-utils.url = "github:numtide/flake-utils";
    # mach-nix.url = "mach-nix/3.5.0";
  };

  outputs = { self, nixpkgs, flake-utils }:
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
          version = "0.5.1";
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
          ];
          cmakeFlags = [
            "-GNinja"
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
            "-DCMAKE_PREFIX_PATH=${pkgs.gtest}"
          ];
          installPhase = parent.installPhase + ''
            install -m555 -D $src/example/sample.aedat4 $out/example/sample.aedat4
            install -m555 -D $src/example/sample.dat $out/example/sample.dat
            install -m755 -D test/aestream_test $out/bin/aestream_test
          '';
        });
        # aestream-python = mach-nix.lib.${system}.buildPythonPackage {
        #   pname = "aestream";
        #   version = "0.5.1";
        #   src = ./.;
        #   requirements = "scikit-build\nnumpy\nnanobind";
        #   providers.pip = "wheel";

        #   buildInputs = [ pkgs.lz4 pkgs.zlib py.pybind11 libcaer ];
        #   nativeBuildInputs = [
        #     pkgs.cmake 
        #     pkgs.which
        #   ];
        #   python = "python310";
        # };
      in
      rec {
        devShells = flake-utils.lib.flattenTree {
          default = aestream;
          # python = aestream-python;
        };
        packages = flake-utils.lib.flattenTree {
          default = aestream;
          test = aestream-test;
        };
      }
    );
}
