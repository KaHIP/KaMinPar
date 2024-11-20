{
  description = "A Shared-Memory and Distributed-Memory Parallel Graph Partitioner";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }: flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = import nixpkgs { inherit system; };

      inputs = builtins.attrValues {
        inherit (pkgs) cmake ninja python3 gcc14 tbb_2021_11 sparsehash mpi numactl pkg-config;
        inherit (pkgs.llvmPackages_19) openmp;
        inherit mt-kahypar;
      };

      devShellInputs = builtins.attrValues {
        inherit (pkgs) fish ccache mold-wrapped gdb;
      };

      mt-kahypar = pkgs.stdenv.mkDerivation {
        pname = "Mt-KaHyPar";
        version = "1.4";

        src = pkgs.fetchFromGitHub {
          owner = "kahypar";
          repo = "mt-kahypar";
          rev = "c51ffeaa3b1040530bf821b7f323e3790b147b33";

          fetchSubmodules = true;
          hash = "sha256-MlF6ZGsqtGQxzDJHbvo5uFj+6w8ehr9V4Ul5oBIGzws=";
        };

        nativeBuildInputs = builtins.attrValues {
          inherit (pkgs) cmake ninja python3 gcc14 boost tbb_2021_11 hwloc;
        };

        cmakeFlags = [
          # The cmake package does not handle absolute CMAKE_INSTALL_INCLUDEDIR
          # correctly (setting it to an absolute path causes include files to go to
          # $out/$out/include, because the absolute path is interpreted with root
          # at $out).
          # See: https://github.com/NixOS/nixpkgs/issues/144170
          "-DCMAKE_INSTALL_INCLUDEDIR=include"
          "-DCMAKE_INSTALL_LIBDIR=lib"
        ];
        enableParallelBuilding = true;

        meta = {
          description = "A shared-memory multilevel graph and hypergraph partitioner.";
          homepage = "https://github.com/kahypar/mt-kahypar";
          license = pkgs.lib.licenses.mit;
        };
      };
    in
    {
      devShells = rec {
        default = gcc;

        gcc = pkgs.mkShell {
          packages = inputs ++ devShellInputs;

          shellHook = ''
            exec fish
          '';
        };

        clang = (pkgs.mkShell.override { stdenv = pkgs.llvmPackages_19.stdenv; }) {
          packages = (pkgs.lib.lists.remove pkgs.gcc14 inputs) ++ devShellInputs;

          shellHook = ''
            exec fish
          '';
        };
      };

      packages.default = pkgs.stdenv.mkDerivation {
        pname = "KaMinPar";
        version = "3.0";

        src = self;
        nativeBuildInputs = inputs;

        cmakeFlags = [
          "-DKAMINPAR_BUILD_DISTRIBUTED=On"
          "-DKAMINPAR_BUILD_WITH_MTKAHYPAR=On"
          "-DKAMINPAR_BUILD_TESTS=Off"
          "-DKAMINPAR_BUILD_WITH_CCACHE=Off"
        ];
        enableParallelBuilding = true;

        meta = {
          description = "A shared-memory and distributed-memory parallel graph partitioner.";
          homepage = "https://github.com/KaHIP/KaMinPar";
          license = pkgs.lib.licenses.mit;
        };
      };
    }
  );
}
