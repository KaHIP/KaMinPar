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
        inherit (pkgs) cmake ninja python3 tbb_2022_0 sparsehash mpi numactl pkg-config;
        inherit (pkgs.llvmPackages_20) openmp;
        inherit mt-kahypar;
      };

      devShellInputs = builtins.attrValues {
        inherit (pkgs) ccache mold-wrapped gdb;
      };

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
            rev = "a4a97ff2b9037c215c533a2889f2eebeb1504662";
            hash = "sha256-6j43kzCEsm/7VEyq3tOEHyQVlBG+uwBAsS0cSBFAp2E=";
          };

          nativeBuildInputs = builtins.attrValues {
            inherit (pkgs) cmake ninja python3 boost tbb_2022_0 hwloc;
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
            "-DCMAKE_INSTALL_INCLUDEDIR=include"
            "-DCMAKE_INSTALL_LIBDIR=lib"
          ];

          meta = {
            description = "A shared-memory multilevel graph and hypergraph partitioner.";
            homepage = "https://github.com/kahypar/mt-kahypar";
            license = pkgs.lib.licenses.mit;
          };
        };
    in
    {
      devShells.default = pkgs.mkShell {
        packages = inputs ++ devShellInputs;
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
