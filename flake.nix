{
  description = "A Shared-Memory and Distributed-Memory Parallel Graph Partitioner";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }: flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = import nixpkgs { inherit system; };
      lib = pkgs.lib;

      kaminpar =
        let
          google-test-src = pkgs.fetchFromGitHub {
            owner = "google";
            repo = "googletest";
            rev = "5a37b517ad4ab6738556f0284c256cae1466c5b4";
            hash = "sha256-uwdRrw79be2N1bBILeVa6q/hzx8MXUG8dcR4DU/cskw=";
          };
        in
        pkgs.stdenv.mkDerivation (finalAttrs: {
          pname = "kaminpar";
          version = "3.5.1";

          src = self;

          doCheck = true;
          strictDeps = true;

          nativeBuildInputs = builtins.attrValues {
            inherit (pkgs) git pkg-config cmake mpi;
          };

          propagatedBuildInputs = builtins.attrValues {
            inherit (pkgs) tbb_2022_0 sparsehash mpi;
            inherit mt-kahypar;
          };

          buildInputs = pkgs.lib.optional pkgs.stdenv.hostPlatform.isLinux pkgs.numactl;

          __darwinAllowLocalNetworking = true;
          nativeCheckInputs = [ pkgs.mpiCheckPhaseHook ];

          cmakeFlags = [
            (lib.cmakeBool "KAMINPAR_BUILD_DISTRIBUTED" true)
            (lib.cmakeBool "KAMINPAR_BUILD_WITH_MTUNE_NATIVE" false)
            (lib.cmakeBool "FETCHCONTENT_FULLY_DISCONNECTED" true)
            (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_KASSERT" "${kassert-src}")
            (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_KAGEN" "${kagen-src}")
            (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_GOOGLETEST" "${google-test-src}")
          ];

          meta = {
            description = "Parallel heuristic solver for the balanced k-way graph partitioning problem";
            homepage = "https://github.com/KaHIP/KaMinPar";
            changelog = "https://github.com/KaHIP/KaMinPar/releases/tag/v${finalAttrs.version}";
            license = lib.licenses.mit;
            platforms = lib.platforms.linux ++ [ "aarch64-darwin" ];
            mainProgram = "KaMinPar";
          };
        });

      mt-kahypar =
        let
          kahypar-shared-resources-src = pkgs.fetchFromGitHub {
            owner = "kahypar";
            repo = "kahypar-shared-resources";
            rev = "6d5c8e2444e4310667ec1925e995f26179d7ee88";
            hash = "sha256-K3tQ9nSJrANdJPf7v/ko2etQLDq2f7Z0V/kvDuWKExM=";
          };

          whfc-src = pkgs.fetchFromGitHub {
            owner = "larsgottesbueren";
            repo = "WHFC";
            rev = "5ae2e3664391ca0db7fab2c82973e98c48937a08";
            hash = "sha256-Oyz6u1uAgQUTOjSWBC9hLbupmQwbzcZJcyxNnj7+qxo=";
          };

          growt-src = pkgs.fetchFromGitHub {
            owner = "TooBiased";
            repo = "growt";
            rev = "0c1148ebcdfd4c04803be79706533ad09cc81d37";
            hash = "sha256-4Vm4EiwmwCs3nyBdRg/MAk8SUWtX6kTukj8gJ7HfJNY=";
          };
        in
        pkgs.stdenv.mkDerivation {
          pname = "Mt-KaHyPar";
          version = "1.5";

          src = pkgs.fetchFromGitHub {
            owner = "kahypar";
            repo = "mt-kahypar";
            rev = "73e11ecbd078382b935fd7c72bb47c23b7afcb57";
            hash = "sha256-TlgFNiwrUQFSXzsGtLBNdZZSIZubN4nn1D6m9VJt1Pw=";
          };

          nativeBuildInputs = builtins.attrValues {
            inherit (pkgs) cmake ninja;
          };

          propagatedBuildInputs = builtins.attrValues {
            inherit (pkgs) tbb_2022_0;
          };

          buildInputs = builtins.attrValues {
            inherit (pkgs) boost hwloc;
          };

          preConfigure = ''
            # Remove the FetchContent_Populate calls in CMakeLists.txt
            sed -i '266,284d' CMakeLists.txt

            # Replace the target_include_directories with custom paths
            substituteInPlace CMakeLists.txt \
              --replace ''\'''${CMAKE_CURRENT_BINARY_DIR}/external_tools/kahypar-shared-resources' '${kahypar-shared-resources-src}'
            substituteInPlace CMakeLists.txt \
              --replace ''\'''${CMAKE_CURRENT_BINARY_DIR}/external_tools/growt' '${growt-src}'
            substituteInPlace CMakeLists.txt \
              --replace ''\'''${CMAKE_CURRENT_BINARY_DIR}/external_tools/WHFC' '${whfc-src}'
          '';

          cmakeFlags = [
            # The cmake package does not handle absolute CMAKE_INSTALL_INCLUDEDIR
            # correctly (setting it to an absolute path causes include files to go to
            # $out/$out/include, because the absolute path is interpreted with root
            # at $out).
            # See: https://github.com/NixOS/nixpkgs/issues/144170
            (lib.cmakeFeature "CMAKE_INSTALL_INCLUDEDIR" "include")
            (lib.cmakeFeature "CMAKE_INSTALL_LIBDIR" "lib")
          ];

          meta = {
            description = "Shared-memory multilevel graph and hypergraph partitioner";
            homepage = "https://github.com/kahypar/mt-kahypar";
            license = lib.licenses.mit;
            platforms = lib.platforms.linux ++ [ "aarch64-darwin" ];
            mainProgram = "mt-kahypar";
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

      kaminpar-python = pkgs.python3Packages.buildPythonPackage {
        pname = "kaminpar";
        version = "3.5.1";

        pyproject = true;
        src = "${self}/bindings/python";

        build-system = builtins.attrValues {
          inherit (pkgs.python3Packages) scikit-build-core pybind11;
        };

        nativeBuildInputs = builtins.attrValues {
          inherit (pkgs) git pkg-config cmake ninja;
        };

        buildInputs = [ pkgs.tbb_2022_0 ] ++ pkgs.lib.optional pkgs.stdenv.hostPlatform.isLinux pkgs.numactl;

        preBuild = pkgs.lib.optionalString pkgs.stdenv.hostPlatform.isDarwin ''
          export CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CXX_COMPILER_AR=$AR -DCMAKE_CXX_COMPILER_RANLIB=$RANLIB"
        '';

        dontUseCmakeConfigure = true;
        CMAKE_ARGS = [
          (lib.cmakeBool "FETCHCONTENT_FULLY_DISCONNECTED" true)
          (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_KAMINPAR" "${self}")
          (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_KASSERT" "${kassert-src}")
        ];

        meta = {
          description = "Python Bindings for KaMinPar";
          homepage = "https://github.com/KaHIP/KaMinPar";
          license = lib.licenses.mit;
          platforms = lib.platforms.linux ++ [ "aarch64-darwin" ];
          mainProgram = "KaMinPar";
        };
      };

      kaminpar-networkit = pkgs.python3Packages.buildPythonPackage {
        pname = "kaminpar-networkit";
        version = "3.5.1";

        pyproject = true;
        src = "${self}/bindings/networkit";

        build-system = builtins.attrValues {
          inherit (pkgs.python3Packages) scikit-build-core cython;
          inherit cython-cmake;
        };

        nativeBuildInputs = builtins.attrValues {
          inherit (pkgs) git pkg-config cmake ninja;
        };

        buildInputs = [ pkgs.tbb_2022_0 pkgs.sparsehash ] ++ pkgs.lib.optional pkgs.stdenv.hostPlatform.isLinux pkgs.numactl;

        dependencies = [ networkit-python ];

        preBuild = pkgs.lib.optionalString pkgs.stdenv.hostPlatform.isDarwin ''
          export CMAKE_ARGS="-DCMAKE_CXX_COMPILER_AR=$AR -DCMAKE_CXX_COMPILER_RANLIB=$RANLIB"
        '';

        dontUseCmakeConfigure = true;
        CMAKE_ARGS = [
          (lib.cmakeBool "FETCHCONTENT_FULLY_DISCONNECTED" true)
          (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_KAMINPAR" "${self}")
          (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_KASSERT" "${kassert-src}")
        ];

        meta = {
          description = "NetworKit Bindings for KaMinPar";
          homepage = "https://github.com/KaHIP/KaMinPar";
          license = lib.licenses.mit;
          platforms = lib.platforms.linux;
          mainProgram = "KaMinPar";
        };
      };

      networkit-python = pkgs.python3Packages.buildPythonPackage {
        pname = "networkit";
        version = "11.1.post1";
        pyproject = true;

        src = pkgs.fetchFromGitHub {
          owner = "networkit";
          repo = "networkit";
          rev = "39ef1be566b82e9afe4d98de3a272d1f255375fb";

          fetchSubmodules = true;
          hash = "sha256-D1scYeu4irbKZu4rtDATwJ6gMWFGPZZ5IO4O/2ZY8fA=";
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
          license = lib.licenses.mit;
          platforms = lib.platforms.linux;
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
          license = lib.licenses.asl20;
        };
      };
    in
    {
      devShells = {
        default = pkgs.mkShell {
          packages = builtins.attrValues {
            # (d)KaMinPar inputs
            inherit (pkgs) git pkg-config cmake tbb_2022_0 sparsehash numactl mpi;
            inherit mt-kahypar;

            # Additional MT-KaHyPar inputs if build from source for KaMinPar
            inherit (pkgs) boost hwloc;

            # Development inputs
            inherit (pkgs) ccache ninja mold-wrapped gdb dpkg rpm;
            inherit (pkgs.python3Packages) build pybind11 ruff mypy;
          };
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
