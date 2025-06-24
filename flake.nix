{
  description = "JAX + CUDA development shell using flakes";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    #nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
        pythonPackages = pkgs.python312Packages;

        python = pkgs.python312.withPackages (ps: with ps; [
          jax
          jaxlib
	  jaxtyping
          #wandb
          equinox
          optax
          notebook
	  tqdm
	  matplotlib
	  seaborn
	  dash
	  scikit-learn
	  pandas
	  pandas-stubs
          #blackjax
        ]);
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            #pkgs.cudatoolkit
            python
          ];
        };
      });
}
