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
        inherit (pkgs) python3 ninja cmake tbb_2021_11 sparsehash mpi numactl pkg-config git;
        inherit (pkgs.llvmPackages_19) openmp;
        inherit (pkgs.python3Packages) pybind11;
        inherit mt-kahypar networkit-python;
      };

      devShellInputs = builtins.attrValues {
        inherit (pkgs) ccache mold-wrapped gdb ruff;
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

      networkit-python = pkgs.python3Packages.buildPythonPackage {
        pname = "networkit";
        version = "11.0";
        pyproject = true;

        src = pkgs.fetchFromGitHub {
          owner = "networkit";
          repo = "networkit";
          rev = "f720ba9678943c771ed04e540946a9ea1f221fd5";

          fetchSubmodules = true;
          hash = "sha256-JsHIkUIVNq8ZwpAzgLkfHna/Xhu12XCpV3M2hQyZZVs=";
        };

        build-system = builtins.attrValues {
          inherit (pkgs.python3Packages) cython setuptools wheel;
        };

        nativeBuildInputs = builtins.attrValues {
          inherit (pkgs) cmake ninja;
        };

        dependencies = builtins.attrValues {
          inherit (pkgs.python3Packages) numpy scipy;
        };
        dontUseCmakeConfigure = true;

        meta = {
          description = "Toolbox for high-performance network analysis";
          homepage = "https://networkit.github.io";
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
          packages = inputs ++ devShellInputs;
        };
      };

      packages.default =
        let
          kassert-src = pkgs.fetchFromGitHub {
            owner = "kamping-site";
            repo = "kassert";
            rev = "988b7d54b79ae6634f2fcc53a0314fb1cf2c6a23";

            fetchSubmodules = true;
            hash = "sha256-CBglUfVl9lgEa1t95G0mG4CCj0OWnIBwk7ep62rwIAA=";
          };

          kagen-src = pkgs.fetchFromGitHub {
            owner = "KarlsruheGraphGeneration";
            repo = "KaGen";
            rev = "70386f48e513051656f020360c482ce6bff9a24f";

            fetchSubmodules = true;
            hash = "sha256-5EvRPpjUZpmAIEgybXjNU/mO0+gsAyhlwbT+syDUr48=";
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
            description = "Shared-memory and distributed-memory parallel graph partitioner";
            homepage = "https://github.com/KaHIP/KaMinPar";
            license = pkgs.lib.licenses.mit;
          };
        };
    }
  );
}
