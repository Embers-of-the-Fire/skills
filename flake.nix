{
  description = "CUDA dev environment";
  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
    in
    {
      devShells.${system}.default =
        (pkgs.buildFHSEnv {
          name = "nvidia-cuda-dev";
          targetPkgs =
            pkgs:
            (with pkgs; [
              zlib

              cudatoolkit
              linuxPackages.nvidia_x11
              libGLU
              libGL
              libxi
              libxmu
              libxext
              libx11
              libxv
              libxrandr

              stdenv.cc
              binutils

              fish
            ]);

          profile = ''
            export LD_LIBRARY_PATH="${pkgs.linuxPackages.nvidia_x11}/lib"
            export CUDA_PATH="${pkgs.cudatoolkit}"
            export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
            export EXTRA_CCFLAGS="-I/usr/include"
          '';

          runScript = "fish";
        }).env;
    };
}
