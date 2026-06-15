#FORCES_MPC_GP_MAP0_DynGP_5 : A fast customized optimization solver.
#
#Copyright (C) 2013-2025 EMBOTECH AG [info@embotech.com]. All rights reserved.
#
#
#This software is intended for simulation and testing purposes only. 
#Use of this software for any commercial purpose is prohibited.
#
#This program is distributed in the hope that it will be useful.
#EMBOTECH makes NO WARRANTIES with respect to the use of the software 
#without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
#PARTICULAR PURPOSE. 
#
#EMBOTECH shall not have any liability for any damage arising from the use
#of the software.
#
#This Agreement shall exclusively be governed by and interpreted in 
#accordance with the laws of Switzerland, excluding its principles
#of conflict of laws. The Courts of Zurich-City shall have exclusive 
#jurisdiction in case of any dispute.
#
import os
import sys
import glob
from distutils.ccompiler import new_compiler
from distutils import unixccompiler

c = new_compiler()

# determine source file
sourcedir = os.path.join(os.getcwd(), "FORCES_MPC_GP_MAP0_DynGP_5", "src")
sourcefile_list = glob.glob(os.path.join(sourcedir, "*.c"))

# determine lib file
libdir = os.path.join(os.getcwd(), "FORCES_MPC_GP_MAP0_DynGP_5", "lib")
lib_opts = []
if sys.platform.startswith('win'):
	static_libformat = "FORCES_MPC_GP_MAP0_DynGP_5_static"
	shared_libformat = "FORCES_MPC_GP_MAP0_DynGP_5"
	lib_opts.append("/DLL")
else:
	static_libformat = "FORCES_MPC_GP_MAP0_DynGP_5"
	shared_libformat = "FORCES_MPC_GP_MAP0_DynGP_5"

# create lib dir if it does not exist yet
if not os.path.exists(libdir):
	os.makedirs(libdir)
				
# compile into object file
objdir = os.path.join(os.getcwd(),"FORCES_MPC_GP_MAP0_DynGP_5","obj")
rel_sourcefile_list = [os.path.relpath(sourcefile, sourcedir) for sourcefile in sourcefile_list]
if not os.path.exists(objdir):
	os.makedirs(objdir)

cur_dir = os.getcwd()
try:
	os.chdir(sourcedir)
	if isinstance(c,unixccompiler.UnixCCompiler):
		objects = c.compile(rel_sourcefile_list, output_dir=objdir, extra_preargs=['-O3','-fPIC', '-fopenmp','-mavx'])
		if sys.platform.startswith('linux'):
			c.set_libraries(['rt','gomp'])
	else:
		objects = c.compile(rel_sourcefile_list, output_dir=objdir)
except Exception as e:
	os.chdir(cur_dir)
	raise(e)
os.chdir(cur_dir)

rel_objects = [os.path.relpath(objfile, objdir) for objfile in objects]
# create libraries
exportsymbols = ["FORCES_MPC_GP_MAP0_DynGP_5_solve", "FORCES_MPC_GP_MAP0_DynGP_5_internal_mem"]
try:
	os.chdir(objdir)
	c.create_static_lib(objects, output_libname=static_libformat, output_dir=libdir)
	c.link_shared_lib(objects, output_libname=shared_libformat, output_dir=libdir, export_symbols=exportsymbols, extra_preargs=lib_opts)
except Exception as e:
	os.chdir(cur_dir)
	raise(e)
os.chdir(cur_dir)
