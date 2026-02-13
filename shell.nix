{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  packages = with pkgs; [
    texliveFull
    texlab # LaTeX language server
  ];
}
