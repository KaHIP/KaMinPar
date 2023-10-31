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
        inherit (pkgs) cmake ninja python312 gcc13 tbb_2021_8 sparsehash mpi;
      };
    in
    {
      devShells.default = pkgs.mkShell {
        packages = inputs ++ builtins.attrValues {
          inherit (pkgs) fish ccache valgrind massif-visualizer;
        };

        shellHook = ''
          exec fish
        '';
      };

      packages.default = pkgs.stdenv.mkDerivation {
        pname = "KaMinPar";
        version = "2.1.0";

        src = self;
        nativeBuildInputs = inputs;

        cmakeFlags = [ "-DKAMINPAR_BUILD_DISTRIBUTED=On" ];
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
