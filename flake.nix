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
      stdenv = pkgs.stdenv;

      kaminpar = stdenv.mkDerivation (finalAttrs: {
        pname = "kaminpar";
        version = "3.6.0";

        src = self;
        strictDeps = true;

        nativeBuildInputs = builtins.attrValues {
          inherit (pkgs) cmake mpi;
        };

        buildInputs = [ pkgs.gtest ] ++ lib.optional stdenv.hostPlatform.isLinux pkgs.numactl;

        propagatedBuildInputs = builtins.attrValues {
          inherit (pkgs) mpi sparsehash tbb_2022_0;
          inherit kagen kassert mt-kahypar;
        };

        cmakeFlags = [
          (lib.cmakeBool "KAMINPAR_BUILD_DISTRIBUTED" true)
          (lib.cmakeBool "KAMINPAR_BUILD_WITH_MTUNE_NATIVE" false)
        ];

        doCheck = true;
        __darwinAllowLocalNetworking = true;
        nativeCheckInputs = [ pkgs.mpiCheckPhaseHook ];

        meta = {
          description = "Parallel heuristic solver for the balanced k-way graph partitioning problem";
          homepage = "https://github.com/KaHIP/KaMinPar";
          changelog = "https://github.com/KaHIP/KaMinPar/releases/tag/v${finalAttrs.version}";
          mainProgram = "KaMinPar";
          license = lib.licenses.mit;
          platforms = lib.platforms.unix;
        };
      });

      kagen = stdenv.mkDerivation (finalAttrs: {
        pname = "kagen";
        version = "1.1.0";

        # TODO: Update to main repo when installation fix is merged
        src = pkgs.fetchFromGitHub {
          owner = "dsalwasser";
          repo = "KaGen";
          rev = "8f810facc264105db3dbd8cd7b2c6b812096dbc1";

          fetchSubmodules = true;
          hash = "sha256-rG7cTsVvr2gSrZGdvxXo7MHCGFGJD2Ih70XB2NuqE6I=";
        };

        nativeBuildInputs = builtins.attrValues {
          inherit (pkgs) cmake pkg-config;
        };

        buildInputs = builtins.attrValues {
          inherit (pkgs) imagemagick;
        };

        propagatedBuildInputs = builtins.attrValues {
          inherit (pkgs) boost cgal gmp mpfr mpi sparsehash;
        };

        cmakeFlags = [
          (lib.cmakeBool "KAGEN_BUILD_EXAMPLES" false)
          (lib.cmakeBool "KAGEN_BUILD_TESTS" finalAttrs.finalPackage.doCheck)
          (lib.cmakeBool "KAGEN_USE_BUNDLED_GTEST" false)
        ];

        doCheck = false;
        __darwinAllowLocalNetworking = true;
        nativeCheckInputs = with pkgs; [
          gtest
          ctestCheckHook
          mpiCheckPhaseHook
        ];

        disabledTests = lib.optionals (stdenv.hostPlatform.isDarwin && stdenv.hostPlatform.isAarch64) [
          # flaky tests on aarch64-darwin
          "test_rgg2d.2cores"
          "test_rgg2d.4cores"
        ];

        meta = {
          description = "Communication-free Massively Distributed Graph Generators";
          homepage = "https://github.com/KarlsruheGraphGeneration/KaGen";
          changelog = "https://github.com/KarlsruheGraphGeneration/KaGen/releases/tag/v${finalAttrs.version}";
          mainProgram = "KaGen";
          license = with lib.licenses; [
            bsd2
            mit
            lib.licenses.boost
          ];
          platforms = lib.platforms.unix;
        };
      });

      kassert = stdenv.mkDerivation (finalAttrs: {
        pname = "kassert";
        version = "0.2.2";

        src = pkgs.fetchFromGitHub {
          owner = "kamping-site";
          repo = "kassert";
          tag = "v${finalAttrs.version}";
          hash = "sha256-5UndFUhKtHPFPLfYP0EI/r+eoAptcQBheznALfxh27s=";
        };

        nativeBuildInputs = [ pkgs.cmake ];

        cmakeFlags = [
          # doc generation require git clone doxygen-awesome-css
          (lib.cmakeBool "KASSERT_BUILD_DOCS" false)
          (lib.cmakeBool "KASSERT_BUILD_TESTS" finalAttrs.finalPackage.doCheck)
          (lib.cmakeBool "KASSERT_USE_BUNDLED_GTEST" false)
        ];

        doCheck = true;
        nativeCheckInputs = [ pkgs.gtest ];

        meta = {
          description = "Karlsruhe assertion library for C++";
          homepage = "https://kamping-site.github.io/kassert/";
          changelog = "https://github.com/kamping-site/kasser/releases/tag/v${finalAttrs.version}";
          license = with lib.licenses; [ mit ];
          platforms = lib.platforms.unix;
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
        stdenv.mkDerivation {
          pname = "Mt-KaHyPar";
          version = "1.5.1";

          src = pkgs.fetchFromGitHub {
            owner = "kahypar";
            repo = "mt-kahypar";
            rev = "8d90c765a0a9f81b917bffab84cb5e3ab45c082b";
            hash = "sha256-2USu34LV60boup+hDftMPpAWdrFyimZA6q5Rx40xW7s=";
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

          nativeBuildInputs = builtins.attrValues {
            inherit (pkgs) cmake ninja;
          };

          buildInputs = builtins.attrValues {
            inherit (pkgs) boost hwloc;
          };

          propagatedBuildInputs = builtins.attrValues {
            inherit (pkgs) tbb_2022_0;
          };

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
            platforms = lib.platforms.unix;
          };
        };

      kaminpar-python = pkgs.python3Packages.buildPythonPackage {
        pname = "kaminpar";
        version = "3.6.0";

        pyproject = true;
        src = "${self}/bindings/python";

        build-system = builtins.attrValues {
          inherit (pkgs.python3Packages) scikit-build-core pybind11;
        };

        nativeBuildInputs = builtins.attrValues {
          inherit (pkgs) git pkg-config cmake ninja;
        };

        preBuild = lib.optionalString stdenv.hostPlatform.isDarwin ''
          export CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CXX_COMPILER_AR=$AR -DCMAKE_CXX_COMPILER_RANLIB=$RANLIB"
        '';

        buildInputs = [ kassert pkgs.tbb_2022_0 ] ++ lib.optional stdenv.hostPlatform.isLinux pkgs.numactl;

        dontUseCmakeConfigure = true;
        CMAKE_ARGS = [
          (lib.cmakeBool "FETCHCONTENT_FULLY_DISCONNECTED" true)
          (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_KAMINPAR" "${self}")
        ];

        meta = {
          description = "Python Bindings for KaMinPar";
          homepage = "https://github.com/KaHIP/KaMinPar";
          license = lib.licenses.mit;
          platforms = lib.platforms.unix;
        };
      };

      kaminpar-networkit = pkgs.python3Packages.buildPythonPackage {
        pname = "kaminpar-networkit";
        version = "3.6.0";

        pyproject = true;
        src = "${self}/bindings/networkit";

        build-system = builtins.attrValues {
          inherit (pkgs.python3Packages) scikit-build-core cython;
          inherit cython-cmake;
        };

        nativeBuildInputs = builtins.attrValues {
          inherit (pkgs) git pkg-config cmake ninja;
        };

        preBuild = lib.optionalString stdenv.hostPlatform.isDarwin ''
          export CMAKE_ARGS="-DCMAKE_CXX_COMPILER_AR=$AR -DCMAKE_CXX_COMPILER_RANLIB=$RANLIB"
        '';

        buildInputs = [ pkgs.tbb_2022_0 pkgs.sparsehash ] ++ lib.optional stdenv.hostPlatform.isLinux pkgs.numactl;

        dependencies = [ kassert networkit-python ];

        dontUseCmakeConfigure = true;
        CMAKE_ARGS = [
          (lib.cmakeBool "FETCHCONTENT_FULLY_DISCONNECTED" true)
          (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_KAMINPAR" "${self}")
        ];

        meta = {
          description = "NetworKit Bindings for KaMinPar";
          homepage = "https://github.com/KaHIP/KaMinPar";
          license = lib.licenses.mit;
          platforms = lib.platforms.unix;
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
            inherit (pkgs) cmake gtest mpi numactl sparsehash tbb_2022_0;
            inherit kagen kassert mt-kahypar;

            # Additional Mt-KaHyPar inputs if build from source for KaMinPar
            inherit (pkgs) boost hwloc;

            # Development inputs
            inherit (pkgs) ccache ninja mold-wrapped gdb dpkg rpm pre-commit gersemi;
            inherit (pkgs.llvmPackages_20) clang-tools;
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
