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
        inherit (pkgs) python3 gcc14 ninja cmake tbb_2021_11 sparsehash mpi numactl pkg-config git;
        inherit (pkgs.llvmPackages_19) openmp;
        inherit mt-kahypar;
      };

      devShellInputs = builtins.attrValues {
        inherit (pkgs) ccache mold-wrapped gdb;
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
        };

        clang = (pkgs.mkShell.override { stdenv = pkgs.llvmPackages_19.stdenv; }) {
          packages = (pkgs.lib.lists.remove pkgs.gcc14 inputs) ++ devShellInputs;
        };
      };

      packages.default =
        let
          mkSourceDerivation = { pname, version, owner, repo, rev, hash, patches }:
            pkgs.stdenvNoCC.mkDerivation {
              inherit pname version;

              src = pkgs.fetchFromGitHub {
                inherit owner repo rev hash;
                fetchSubmodules = true;
              };
              patches = patches;

              installPhase = ''
                runHook preInstall

                mkdir -p $out
                cp -r . $out

                runHook postInstall
              '';
            };

          kassert-src = mkSourceDerivation {
            pname = "kassert-src";
            version = "0.1.0";

            owner = "kamping-site";
            repo = "kassert";
            rev = "a1aa9eb4ece3b57ef7d989aa56fd1f64162d5ea0";
            hash = "sha256-EGfsGclAFDaPTqAXCrRgqUeHT2HYpV/ZAaRq8LbsxoI=";

            patches = [ "${self}/scripts/KAssert.patch" ];
          };

          kagen-src = mkSourceDerivation {
            pname = "kagen-src";
            version = "1.0.3";

            owner = "KarlsruheGraphGeneration";
            repo = "KaGen";
            rev = "bda1e9718a4e91f81c922d566a47acfbc35ecf54";
            hash = "sha256-2FYeWW1IMxV9mxlIRzv/Boi02TqeJDNSwEynkl5AelI=";

            patches = [ "${self}/scripts/KaGen.patch" ];
          };
        in
        pkgs.stdenv.mkDerivation {
          pname = "KaMinPar";
          version = "3.1.0";

          src = self;
          nativeBuildInputs = inputs;

          cmakeFlags = [
            "-DKAMINPAR_BUILD_DISTRIBUTED=On"
            "-DFETCHCONTENT_FULLY_DISCONNECTED=On"
            "-DFETCHCONTENT_SOURCE_DIR_KASSERT=${kassert-src}"
            "-DFETCHCONTENT_SOURCE_DIR_KAGEN=${kagen-src}"
          ];

          meta = {
            description = "A shared-memory and distributed-memory parallel graph partitioner";
            homepage = "https://github.com/KaHIP/KaMinPar";
            license = pkgs.lib.licenses.mit;
          };
        };
    }
  );
}
