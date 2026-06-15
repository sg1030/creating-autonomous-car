import numpy
import ctypes

name = "FORCES_MPC_GP_MAP0_DynGP_5"
requires_callback = True
lib = "lib/libFORCES_MPC_GP_MAP0_DynGP_5.so"
lib_static = "lib/libFORCES_MPC_GP_MAP0_DynGP_5.a"
c_header = "include/FORCES_MPC_GP_MAP0_DynGP_5.h"
nstages = 20

# Parameter             | Type    | Scalar type      | Ctypes type    | Numpy type   | Shape     | Len
params = \
[("xinit"               , "dense" , ""               , ctypes.c_double, numpy.float64, (  6,   1),    6),
 ("x0"                  , "dense" , ""               , ctypes.c_double, numpy.float64, (160,   1),  160),
 ("all_parameters"      , "dense" , ""               , ctypes.c_double, numpy.float64, (160,   1),  160)]

# Output                | Type    | Ctypes type    | Numpy type   | Shape     | Len
outputs = \
[("x01"                 , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x02"                 , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x03"                 , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x04"                 , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x05"                 , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x06"                 , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x07"                 , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x08"                 , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x09"                 , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x10"                 , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x11"                 , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x12"                 , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x13"                 , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x14"                 , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x15"                 , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x16"                 , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x17"                 , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x18"                 , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x19"                 , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x20"                 , ""               , ctypes.c_double, numpy.float64,     (  8,),    8)]

# Info Struct Fields
info = \
[('it', ctypes.c_int),
 ('it2opt', ctypes.c_int),
 ('res_eq', ctypes.c_double),
 ('res_ineq', ctypes.c_double),
 ('rsnorm', ctypes.c_double),
 ('rcompnorm', ctypes.c_double),
 ('pobj', ctypes.c_double),
 ('dobj', ctypes.c_double),
 ('dgap', ctypes.c_double),
 ('rdgap', ctypes.c_double),
 ('mu', ctypes.c_double),
 ('mu_aff', ctypes.c_double),
 ('sigma', ctypes.c_double),
 ('lsit_aff', ctypes.c_int),
 ('lsit_cc', ctypes.c_int),
 ('step_aff', ctypes.c_double),
 ('step_cc', ctypes.c_double),
 ('solvetime', ctypes.c_double),
 ('fevalstime', ctypes.c_double),
 ('solver_id', ctypes.c_int * 8)
]

# Dynamics dimensions
#   nvar    |   neq   |   dimh    |   dimp    |   diml    |   dimu    |   dimhl   |   dimhu    
dynamics_dims = [
	(8, 6, 0, 8, 6, 6, 0, 0), 
	(8, 6, 0, 8, 6, 6, 0, 0), 
	(8, 6, 0, 8, 6, 6, 0, 0), 
	(8, 6, 0, 8, 6, 6, 0, 0), 
	(8, 6, 0, 8, 6, 6, 0, 0), 
	(8, 6, 0, 8, 6, 6, 0, 0), 
	(8, 6, 0, 8, 6, 6, 0, 0), 
	(8, 6, 0, 8, 6, 6, 0, 0), 
	(8, 6, 0, 8, 6, 6, 0, 0), 
	(8, 6, 0, 8, 6, 6, 0, 0), 
	(8, 6, 0, 8, 6, 6, 0, 0), 
	(8, 6, 0, 8, 6, 6, 0, 0), 
	(8, 6, 0, 8, 6, 6, 0, 0), 
	(8, 6, 0, 8, 6, 6, 0, 0), 
	(8, 6, 0, 8, 6, 6, 0, 0), 
	(8, 6, 0, 8, 6, 6, 0, 0), 
	(8, 6, 0, 8, 6, 6, 0, 0), 
	(8, 6, 0, 8, 6, 6, 0, 0), 
	(8, 6, 0, 8, 6, 6, 0, 0), 
	(8, 6, 0, 8, 6, 6, 0, 0)
]