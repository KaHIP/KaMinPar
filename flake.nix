{
  description = "Shared-memory and distributed graph partitioner for large k partitioning.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }: flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = import nixpkgs { inherit system; };

      inputs = builtins.attrValues {
        inherit (pkgs) cmake ninja python312 gcc14 tbb_2021_11 sparsehash mpi;
        inherit (pkgs.llvmPackages_18) openmp;
      };

      devShellInputs = builtins.attrValues {
        inherit (pkgs) fish ccache gdb;
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

        clang = (pkgs.mkShell.override { stdenv = pkgs.llvmPackages_18.stdenv; }) {
          packages = (pkgs.lib.lists.remove pkgs.gcc14 inputs) ++ devShellInputs;

          shellHook = ''
            exec fish
          '';
        };
      };

      packages.default = pkgs.stdenv.mkDerivation {
        pname = "KaMinPar";
        version = "2.1.0";

        src = self;
        nativeBuildInputs = inputs;

        cmakeFlags = [ "-DKAMINPAR_BUILD_DISTRIBUTED=On" "-DKAMINPAR_BUILD_TESTS=Off" ];
        enableParallelBuilding = true;

        meta = {
          description = "Shared-memory and distributed graph partitioner for large k partitioning.";
          homepage = "https://github.com/KaHIP/KaMinPar";
          license = pkgs.lib.licenses.mit;
        };
      };
    }
  );
}
