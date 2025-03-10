{
  description = "A Shared-Memory and Distributed-Memory Parallel Graph Partitioner";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }: flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = import nixpkgs { inherit system; };

      kaminparInputs = builtins.attrValues {
        inherit (pkgs) python3 ninja cmake tbb_2021_11 sparsehash numactl pkg-config;
        inherit mt-kahypar;
      };

      dkaminparInputs = builtins.attrValues {
        inherit (pkgs) mpi;
        inherit (pkgs.llvmPackages_19) openmp;
      };

      devShellInputs = builtins.attrValues {
        inherit (pkgs) ccache mold-wrapped gdb ruff;
      };

      kaminpar = pkgs.stdenv.mkDerivation {
        pname = "KaMinPar";
        version = "3.1.0";

        src = self;
        nativeBuildInputs = kaminparInputs ++ dkaminparInputs;

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

      kaminpar-python =
        let
          # TODO: use $self when merged
          upstream-kaminpar-src = pkgs.fetchFromGitHub {
            owner = "dsalwasser";
            repo = "KaMinPar";
            rev = "a6dd12fb212e235583ab602f8be1c4e31d190ac7";
            hash = "sha256-RK8sPFxSyiyPzx6SsqM9ubAFKeEUGtDQC8qOYjZb2+Y=";
          };
        in
        pkgs.python3Packages.buildPythonPackage {
          pname = "kaminpar";
          version = "3.1.0";
          pyproject = true;

          src = "${self}/bindings/python";

          build-system = builtins.attrValues {
            inherit (pkgs.python3Packages) scikit-build-core pybind11;
          };

          nativeBuildInputs = builtins.attrValues {
            inherit (pkgs) cmake ninja;
          };

          dependencies = kaminparInputs;

          dontUseCmakeConfigure = true;
          CMAKE_ARGS = [
            "-DFETCHCONTENT_FULLY_DISCONNECTED=On"
            "-DFETCHCONTENT_SOURCE_DIR_KAMINPAR=${upstream-kaminpar-src}"
            "-DFETCHCONTENT_SOURCE_DIR_KASSERT=${kassert-src}"
          ];

          meta = {
            description = "Python Bindings for KaMinPar";
            homepage = "https://github.com/KaHIP/KaMinPar";
            license = pkgs.lib.licenses.mit;
          };
        };

      kaminpar-networkit =
        let
          # TODO: use $self when merged
          upstream-kaminpar-src = pkgs.fetchFromGitHub {
            owner = "dsalwasser";
            repo = "KaMinPar";
            rev = "a6dd12fb212e235583ab602f8be1c4e31d190ac7";
            hash = "sha256-RK8sPFxSyiyPzx6SsqM9ubAFKeEUGtDQC8qOYjZb2+Y=";
          };
        in
        pkgs.python3Packages.buildPythonPackage {
          pname = "kaminpar-networkit";
          version = "3.1.0";
          pyproject = true;

          src = "${self}/bindings/networkit";

          build-system = builtins.attrValues {
            inherit (pkgs.python3Packages) scikit-build-core cython;
            inherit cython-cmake;
          };

          nativeBuildInputs = builtins.attrValues {
            inherit (pkgs) cmake ninja;
          };

          dependencies = kaminparInputs ++ [ networkit-python ];

          dontUseCmakeConfigure = true;
          CMAKE_ARGS = [
            "-DFETCHCONTENT_FULLY_DISCONNECTED=On"
            "-DFETCHCONTENT_SOURCE_DIR_KAMINPAR=${upstream-kaminpar-src}"
            "-DFETCHCONTENT_SOURCE_DIR_KASSERT=${kassert-src}"
          ];

          meta = {
            description = "NetworKit Bindings for KaMinPar";
            homepage = "https://github.com/KaHIP/KaMinPar";
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

      cython-cmake = pkgs.python3Packages.buildPythonPackage {
        pname = "cython-cmake";
        version = "2.1.0";
        pyproject = true;

        src = pkgs.fetchFromGitHub {
          owner = "scikit-build";
          repo = "cython-cmake";
          rev = "aef4790b3e896f505fdcb378f8570b5c18b5e335";
          hash = "sha256-Rr5PaVx+zzYyC1dWOH4nuMSfqKrCALrRptXINp0mNrk=";
        };

        build-system = builtins.attrValues {
          inherit (pkgs.python3Packages) hatchling hatch-vcs cython;
        };

        meta = {
          description = "CMake helpers for building Cython modules";
          homepage = "https://github.com/scikit-build/cython-cmake";
          license = pkgs.lib.licenses.asl20;
        };
      };
    in
    {
      devShells = {
        default = pkgs.mkShell {
          packages = kaminparInputs ++ dkaminparInputs ++ devShellInputs;
        };

        python = pkgs.mkShell {
          packages = [ kaminpar-python kaminpar-networkit ];
        };
      };

      packages = {
        default = kaminpar;

        inherit kaminpar kaminpar-python kaminpar-networkit;
      };
    }
  );
}
