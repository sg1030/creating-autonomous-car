/*
 * AD tool to FORCESPRO Template - missing information to be filled in by createADTool.m 
 * (C) embotech AG, Zurich, Switzerland, 2013-2025. All rights reserved.
 *
 * This file is part of the FORCESPRO client, and carries the same license.
 */ 

#ifdef __cplusplus
extern "C" {
#endif

#include "include/FORCES_MPC_GP_MAP0_DynGP_8.h"

#ifndef NULL
#define NULL ((void *) 0)
#endif

#include "FORCES_MPC_GP_MAP0_DynGP_8_model.h"



/* copies data from sparse matrix into a dense one */
static void FORCES_MPC_GP_MAP0_DynGP_8_sparse2fullcopy(solver_int32_default nrow, solver_int32_default ncol, const solver_int32_default *colidx, const solver_int32_default *row, FORCES_MPC_GP_MAP0_DynGP_8_callback_float *data, FORCES_MPC_GP_MAP0_DynGP_8_float *out)
{
    solver_int32_default i, j;
    
    /* copy data into dense matrix */
    for(i=0; i<ncol; i++)
    {
        for(j=colidx[i]; j<colidx[i+1]; j++)
        {
            out[i*nrow + row[j]] = ((FORCES_MPC_GP_MAP0_DynGP_8_float) data[j]);
        }
    }
}




/* AD tool to FORCESPRO interface */
extern solver_int32_default FORCES_MPC_GP_MAP0_DynGP_8_adtool2forces(FORCES_MPC_GP_MAP0_DynGP_8_float *x,        /* primal vars                                         */
                                 FORCES_MPC_GP_MAP0_DynGP_8_float *y,        /* eq. constraint multiplers                           */
                                 FORCES_MPC_GP_MAP0_DynGP_8_float *l,        /* ineq. constraint multipliers                        */
                                 FORCES_MPC_GP_MAP0_DynGP_8_float *p,        /* parameters                                          */
                                 FORCES_MPC_GP_MAP0_DynGP_8_float *f,        /* objective function (scalar)                         */
                                 FORCES_MPC_GP_MAP0_DynGP_8_float *nabla_f,  /* gradient of objective function                      */
                                 FORCES_MPC_GP_MAP0_DynGP_8_float *c,        /* dynamics                                            */
                                 FORCES_MPC_GP_MAP0_DynGP_8_float *nabla_c,  /* Jacobian of the dynamics (column major)             */
                                 FORCES_MPC_GP_MAP0_DynGP_8_float *h,        /* inequality constraints                              */
                                 FORCES_MPC_GP_MAP0_DynGP_8_float *nabla_h,  /* Jacobian of inequality constraints (column major)   */
                                 FORCES_MPC_GP_MAP0_DynGP_8_float *hess,     /* Hessian (column major)                              */
                                 solver_int32_default stage,     /* stage number (0 indexed)                           */
                                 solver_int32_default iteration, /* iteration number of solver                         */
                                 solver_int32_default threadID   /* Id of caller thread                                */)
{
    /* AD tool input and output arrays */
    const FORCES_MPC_GP_MAP0_DynGP_8_callback_float *in[4];
    FORCES_MPC_GP_MAP0_DynGP_8_callback_float *out[7];
    

    /* Allocate working arrays for AD tool */
    
    FORCES_MPC_GP_MAP0_DynGP_8_callback_float w[584];
	
    /* temporary storage for AD tool sparse output */
    FORCES_MPC_GP_MAP0_DynGP_8_callback_float this_f = (FORCES_MPC_GP_MAP0_DynGP_8_callback_float) 0.0;
    FORCES_MPC_GP_MAP0_DynGP_8_float nabla_f_sparse[2];
    
    
    FORCES_MPC_GP_MAP0_DynGP_8_float c_sparse[6];
    FORCES_MPC_GP_MAP0_DynGP_8_float nabla_c_sparse[37];
    
    
    /* pointers to row and column info for 
     * column compressed format used by AD tool */
    solver_int32_default nrow, ncol;
    const solver_int32_default *colind, *row;
    
    /* set inputs for AD tool */
    in[0] = x;
    in[1] = p;
    in[2] = l;
    in[3] = y;

	if ((0 <= stage && stage <= 18))
	{
		
		
		out[0] = &this_f;
		out[1] = nabla_f_sparse;
		FORCES_MPC_GP_MAP0_DynGP_8_objective_0(in, out, NULL, w, 0);
		if( nabla_f != NULL )
		{
			nrow = FORCES_MPC_GP_MAP0_DynGP_8_objective_0_sparsity_out(1)[0];
			ncol = FORCES_MPC_GP_MAP0_DynGP_8_objective_0_sparsity_out(1)[1];
			colind = FORCES_MPC_GP_MAP0_DynGP_8_objective_0_sparsity_out(1) + 2;
			row = FORCES_MPC_GP_MAP0_DynGP_8_objective_0_sparsity_out(1) + 2 + (ncol + 1);
				
			FORCES_MPC_GP_MAP0_DynGP_8_sparse2fullcopy(nrow, ncol, colind, row, nabla_f_sparse, nabla_f);
		}
		
		out[0] = c_sparse;
		out[1] = nabla_c_sparse;
		FORCES_MPC_GP_MAP0_DynGP_8_dynamics_0(in, out, NULL, w, 0);
		if( c != NULL )
		{
			nrow = FORCES_MPC_GP_MAP0_DynGP_8_dynamics_0_sparsity_out(0)[0];
			ncol = FORCES_MPC_GP_MAP0_DynGP_8_dynamics_0_sparsity_out(0)[1];
			colind = FORCES_MPC_GP_MAP0_DynGP_8_dynamics_0_sparsity_out(0) + 2;
			row = FORCES_MPC_GP_MAP0_DynGP_8_dynamics_0_sparsity_out(0) + 2 + (ncol + 1);
				
			FORCES_MPC_GP_MAP0_DynGP_8_sparse2fullcopy(nrow, ncol, colind, row, c_sparse, c);
		}
		if( nabla_c != NULL )
		{
			nrow = FORCES_MPC_GP_MAP0_DynGP_8_dynamics_0_sparsity_out(1)[0];
			ncol = FORCES_MPC_GP_MAP0_DynGP_8_dynamics_0_sparsity_out(1)[1];
			colind = FORCES_MPC_GP_MAP0_DynGP_8_dynamics_0_sparsity_out(1) + 2;
			row = FORCES_MPC_GP_MAP0_DynGP_8_dynamics_0_sparsity_out(1) + 2 + (ncol + 1);
				
			FORCES_MPC_GP_MAP0_DynGP_8_sparse2fullcopy(nrow, ncol, colind, row, nabla_c_sparse, nabla_c);
		}
	}
	if ((19 == stage))
	{
		
		
		out[0] = &this_f;
		out[1] = nabla_f_sparse;
		FORCES_MPC_GP_MAP0_DynGP_8_objective_1(in, out, NULL, w, 0);
		if( nabla_f != NULL )
		{
			nrow = FORCES_MPC_GP_MAP0_DynGP_8_objective_1_sparsity_out(1)[0];
			ncol = FORCES_MPC_GP_MAP0_DynGP_8_objective_1_sparsity_out(1)[1];
			colind = FORCES_MPC_GP_MAP0_DynGP_8_objective_1_sparsity_out(1) + 2;
			row = FORCES_MPC_GP_MAP0_DynGP_8_objective_1_sparsity_out(1) + 2 + (ncol + 1);
				
			FORCES_MPC_GP_MAP0_DynGP_8_sparse2fullcopy(nrow, ncol, colind, row, nabla_f_sparse, nabla_f);
		}
	}
    
    /* add to objective */
    if (f != NULL)
    {
        *f += ((FORCES_MPC_GP_MAP0_DynGP_8_float) this_f);
    }

    return 0;
}

#ifdef __cplusplus
} /* extern "C" */
#endif
