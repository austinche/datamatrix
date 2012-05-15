from distutils.core import setup, Extension

mod = Extension( 'datamatrix',
                 include_dirs = ['/usr/local/include'],
                 library_dirs = ['/usr/local/lib'],
                 sources = ['datamatrix.c', 'decode_rs_char.c', 'init_rs_char.c'] )

setup( name = 'datamatrix',
       version = '0.1',
       description = 'A simple datamatrix decoder',
       py_modules = ['datamatrix'],
       ext_modules = [mod] )
