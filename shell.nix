{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  packages = with pkgs; [
    texliveFull
    texlab # LaTeX language server

    python313
    python313Packages.pip
    uv
    gcc.cc.lib # Numpy support
    zlib

    # Add X11 libs required by OpenCV / libxcb
    libxcb
    libx11
    libxext
    libxrender
    libxtst
    libxfixes
    
    mesa # Provides libGL.so.1
    libGL
    
    glib # libgthread-2.0.so.0
  ];

  # ensure the dynamic linker env points at the nix store libs
  shellHook = ''
    # Compute the nix library path for the needed runtime closures
    export NIX_LD_LIBRARY_PATH=${
      with pkgs;
      lib.makeLibraryPath [
        gcc.cc.lib
        zlib
        libxcb
        libx11
        libxext
        libxrender
        libxtst
        libxfixes
        mesa
        libGL
        glib
      ]
    }

    # Make the dynamic loader also see the nix path (this helps processes started in the shell)
    export LD_LIBRARY_PATH="$NIX_LD_LIBRARY_PATH''${LD_LIBRARY_PATH:+:}''$LD_LIBRARY_PATH"
  '';
}
