# Our derivation here is a simple function.
# The first "{ ... }:" is a set of arguments followed by a colon to denote that it's a function
# Our two arguments are given default values using "?" of:
# - The result of importing the nixpkgs 22.05 release (hopefully ensures reproducibility?)
# - The aarch64-linux-gnu cross compilation subset from within that
#   - The most complex part here is _definitely_ the overlay to patch the papi derivation
#     because we want to have aarch64 libpapi available in our nix shell, but it's hard to cross compile
#     and the default derivation in nixpkg can't handle it by itself. So we define an overlay inline.
# And the "rest" is the body of the function.
# The function body involves running mkShell, a special type of derivation that gives us a development shell
{ 
  # import accepts (i) a thing to import and (ii) arguments to pass that thing being imported -- an empty set in our case
  hostPkgs ? import ( fetchTarball https://github.com/NixOS/nixpkgs/archive/refs/tags/22.05.tar.gz ) {},
  aarch64Pkgs ? import ( fetchTarball https://github.com/NixOS/nixpkgs/archive/refs/tags/22.05.tar.gz ) {
    crossSystem = {
      config = "aarch64-linux-gnu";
    };

    # Because we want to use papi as an aarch64 package (and it needs autoconf flags for cross compilation),
    # Specify them here via an inline overlay.
    overlays = [
      (self: super: 
        # TODO: these flags get it to compile, but they could totally be wrong in practice
        # I just stole them from here: https://stackoverflow.com/a/50528142
        let newConfigureFlags = [ "--with-ffsll"
                                  "--with-virtualtimer=perfctr"
                                  "--with-walltimer=gettimeofday"
                                  "--with-tls=__thread"
                                  "--with-perf-events"
                                  "--with-arch=aarch64"
                                  "--with-CPU=arm" ];
            newCFlags = " -DUSE_PTHREAD_MUTEXES";
        in {
          papi = super.papi.overrideAttrs(oldAttrs: {
            configureFlags = [ ( oldAttrs.configureFlags or [] ) ] ++ newConfigureFlags;
            NIX_CFLAGS_COMPILE = ( oldAttrs.NIX_CFLAGS_COMPILE or "" ) + newCFlags;
          });
        }
      )
    ];
  }
}:

hostPkgs.mkShell {
  name = "CEDR Nix Development Shell";

  packages = [
    hostPkgs.curl
    hostPkgs.unzip
    hostPkgs.papi
    hostPkgs.gnutar
    hostPkgs.vim
    hostPkgs.gdb
    hostPkgs.git
    hostPkgs.gcc
    hostPkgs.gsl
    hostPkgs.cmake
    aarch64Pkgs.gcc
    aarch64Pkgs.gsl
    aarch64Pkgs.papi
  ];

  # Customize the PS1 of this shell
  shellHook = ''
    echo "Entering CEDR Nix Development Shell..."
    export PS1="\[\033[1;32m\][nix-shell:\w]\$\[\033[0m\] "
  '';
}
