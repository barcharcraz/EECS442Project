PATH=~bartoc/install/usr/bin/:~bartoc/install/usr/local/bin/:~bartoc/install/bin:$PATH
PKG_CONFIG_PATH=~bartoc/install/usr/lib/pkgconfig/:~bartoc/install/lib/pkgconfig:$PKG_CONFIG_PATH
LD_LIBRARY_PATH=~bartoc/install/usr/lib:~bartoc/install/lib:$LD_LIBRARY_PATH
LIBRARY_PATH=~bartoc/install/usr/lib:~bartoc/install/lib:$LIBRARY_PATH
C_INCLUDE_PATH=~bartoc/install/include:$C_INCLUDE_PATH
CPLUS_INCLUDE_PATH=~bartoc/install/include:$CPLUS_INCLUDE_PATH
MANPATH=~bartoc/install/share/man:$MANPATH
export PATH
export PKG_CONFIG_PATH
export LD_LIBRARY_PATH
export LIBRARY_PATH
export C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH
export MANPATH
module load cuda
