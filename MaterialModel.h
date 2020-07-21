#ifndef MaterialModel_H
#define MaterialModel_H
/*
 *
 * Author: Johannes Friedlein, 2019/2020; based on Dominic Soldner, 2017
 * 
 * 
 *
 
 * Author: Johannes Friedlein, 2019/2020; based on Dominic Soldner, FAU Erlangen-Nuremberg, 2017
 
 */

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <iostream>

#include "StrainMeasures.h"
#include <deal.II/physics/transformations.h>

// Handling 3D-2D-ax problems
// @todo-optimise: Do we need/use this in here?
#include "../2D_axial-symmetry_plane-strain_dealii/handling_2D.h"

#include "enumerator_list.h"

// Modules for elasto-plasticity
#include "../Elasto-plasticity_modules_dealii/elpl_equation_list.h"

using namespace dealii;


//-----------------------------------------------------------
//-----------------------------------------------------------
//-----------------------------------------------------------
//-----------------------------------------------------------
//-----------------------------------------------------------
//-----------------------------------------------------------




template<int dim>
class MaterialModel
{
    public:
		MaterialModel( /*const //1153*/ Parameter::GeneralParameters &parameter, const std::string &pre_txt_directory,
					   const unsigned int &newton_iteration, const unsigned int &current_load_step, const bool &unloading );
        ~MaterialModel(){}
		
		 /*
		  * ############################################################ Elasto-Plasticity #######################################################
		  */


		 // Elasto-plasticity - 3D
		 void elastoplasticity  ( SymmetricTensor<2,3> &eps_n1_3D, double &alpha_n,
								   SymmetricTensor<2,3> &eps_p_n, unsigned int hardening_type,
								   SymmetricTensor<2,3> &sigma_n1_3D, SymmetricTensor<4,3> &C_ep_3D, SymmetricTensor<2,dim> &Tangent_theta, double &stress_vM,
								   SymmetricTensor<4,dim> &Lambda, bool &GG_mode_requested );

    private:
		Parameter::GeneralParameters &parameter;

	  // Lame and material parameters
		/*! The Lame parameter \f$ \mu \f$
		 */
         double mu;
		/*! The Lame parameter \f$ \lambda \f$
		 */		
		 double lambda;

    public:
   		const double nu;

   		const double kappa = lambda + 2./3. * mu; // bulk modulus; valid for plane strain and 3D

   		enum enum_plasti_dmg
   		{
   			plasti = 0,
   			dmg =    1
   		};

    private:
		/*! The yield stress \f$ \sigma_y \f$
		*/
		double yield_stress;

		/*! The hardening modulus \f$ K \f$
		*/
		const double K;

		// We just use them right now to check whether we need to kick something (corresponding kick_something(*) fnc)
		// in case we want to compute a non-unloading tangent
		const unsigned int newton_iteration, current_load_step;

		// userParameters
		const unsigned int max_nbr_FB_iterations = 20;	// #q: sensible choice needed here
		const double tol = 1e-8; // originally 1e-8	// #q: use a sensible value here
		const bool use_ACTIVE_SET = true;

		const double smallDev = 1e-100; //1e-20

    public:
		double q_min;

		const double beta_inf;

		/*const*/ double beta_d, c_d; // non-constant because of the insurance in the local dmg model: there these variables are set to zero to ensure locality

		const std::string pre_txt_directory;

		const bool unloading;

		enums::enum_hardening_law plastic_hardening;

		bool debugging = false;
};




//Definition of the class template
//-----------------------------------------------------------
//-----------------------------------------------------------
//-----------------------------------------------------------
template <int dim> //Constructor
MaterialModel<dim>::MaterialModel( /*const //1153*/  Parameter::GeneralParameters &parameter, const std::string &pre_txt_directory,
								   const unsigned int &newton_iteration, const unsigned int &current_load_step, const bool &unloading  )
:
parameter ( parameter ),
mu( parameter.lame_mu ),
lambda( parameter.lame_lambda ),
nu ( parameter.nu),
yield_stress( parameter.yield_stress ),
K ( parameter.K ),
eta_i (5), // Just set the size and enter the values below from the parameter set
newton_iteration ( newton_iteration ),
current_load_step ( current_load_step ),
q_min ( parameter.q_min ),
beta_inf ( parameter.beta_inf ),
beta_d ( parameter.beta_d ),
c_d ( parameter.c_d ),
pre_txt_directory( pre_txt_directory ),
unloading ( unloading )
{	
}
//------------------------------------------



// @section Plasti Elasto-Plasticity
/*
 * ############################################################ Elasto-Plasticity #######################################################
 */

// @subsection SStrain_Plasti Elasto-Plasticity - Small strains

// 3D-elasto-plasticity with elasto-plasticity module
template <int dim>
void MaterialModel<dim>::elastoplasticity
						(
							SymmetricTensor<2,3> &eps_n1_3D, double &alpha_n, SymmetricTensor <2,3> &eps_p_n, unsigned int hardening_type,
							SymmetricTensor<2,3> &sigma_n1_3D, SymmetricTensor<4,3> &C_ep_3D,
							SymmetricTensor<2,dim> &Tangent_theta/*here unnecessary but needed to be able to call the same function for 2D and 3D*/,
							double &stress_vM, SymmetricTensor<4,dim> &Lambda, bool &GG_mode_requested
						)
{
	plastic_hardening = enums::enum_hardening_law(hardening_type);
	double kappa = parameter.kappa;
	double sqrt2_3 = std::sqrt(2./3.);

	// Create the elpl module and init it with all necessary variables and parameters
	 elastoplastic_equations<dim,double> eq_list ( hardening_type, mu, kappa, yield_stress, K, beta_inf, parameter.K_exp,
			 	 	 	 	 	 	 	 	 	   eps_n1_3D, eps_p_n, alpha_n, parameter.HillT_H );

	// Compute trial values:
	 SymmetricTensor<2,3,double> sigma_t_n1 = eq_list.get_stress_T_t_ep();

	 double Phi_t_n1 = eq_list.get_plastic_yield_fnc(sigma_t_n1, alpha_n);

	std::ofstream write_residuum(pre_txt_directory+"residuum.txt", std::ios::app);
	if ( parameter.write_quad_points == true )
		write_residuum << 0 << " , " << Phi_t_n1 << " , " << 0  << std::endl;

	// Tangent modulus: elastic part (independent of the hardening law)
	// This elastic contribution is always written into the variable \a C_ep_3D, because in case
	// something goes fubar, the tangent is at least not singular and won't light up the assembly.
	 C_ep_3D = kappa * StandardTensors::IxI<3>()
			   + 2. * mu * StandardTensors::I_deviatoric<3>();

	if ( Phi_t_n1 <= tol ) // numerical zero
	{
		// elastic:
		sigma_n1_3D = sigma_t_n1;
	}
	else
	{
		// plastic:
		double gamma_k = 0.;
		double Phi_k = Phi_t_n1;
		double alpha_k (alpha_n);
		SymmetricTensor<2,3> sigma_n1 = sigma_t_n1;

		double gamma_update = 0.;
		unsigned int k=0; // out of the scope for the convergence check below
		for ( ; k<max_nbr_FB_iterations; k++ )
		{
			// Update Lagrange multiplier increment
			 gamma_update = - Phi_k / eq_list.get_dPhi_dgamma_ep(gamma_k);
			 gamma_k = gamma_k + gamma_update;

			// Update relevant history and quantities for new yield function
			 alpha_k = eq_list.get_alpha_n_k_ep(gamma_k);
			 sigma_n1 = eq_list.get_stress_n1_ep( gamma_k, alpha_k );

			 // The norm of the evolution direction is no longer 1 !
			 eq_list.update_n_n1_ep(sigma_n1); // update only needed for aniso, for isotropy the evolution direction \a n_n1 remains constant

			// Check the yield function (done in the end of the loop, because we already computed Phi_t for the first iteration with gamma_k=0)
			 Phi_k = eq_list.get_plastic_yield_fnc(sigma_n1, alpha_k);

			if ( parameter.write_quad_points == true )
				write_residuum << k+1 << " , " << std::abs(Phi_k) << " , " << gamma_k  << std::endl;

			// Satisfied yield function -> converged -> exit
			 if ( std::abs(Phi_k) < tol ) //needs to be consistent with the above check of the trial yield fnc
				break;
		}

		// Check the max nbr of iterations
		 if ( k >= max_nbr_FB_iterations ) // The iterations failed
		 {
			 if ( parameter.get_going_mode || parameter.damped_NR )
				 GG_mode_requested = true;
			 else
			 {
	 			 std::cout << "eps_n1=" << eps_n1_3D << std::endl;
	 			 std::cout << "alpha_n=" << alpha_n << std::endl;
	 			 std::cout << "alpha_k=" << alpha_k << std::endl;
	 			 std::cout << "eps_p_n=" << eps_p_n << std::endl;
	 			 std::cout << "Phi_k=" << Phi_k << std::endl;
	 			 std::cout << "gamma_k=" << gamma_k << std::endl;
	 			 AssertThrow ( false, ExcMessage("Material model - Elasto-plasticity with saturation<< "
												 "No convergence in the subiterations! This could stem from "
												 "convergence issues or the max number of allowed iterations being too small.") );
			 }
		 }
		 // This else is crucial, else we would also update the history based on the failed iterations
		 else // The iterations were successful
		 {
			// Tangent modulus: add the plastic part for plasticity:
			 C_ep_3D = eq_list.get_tangent_plastic_ep(sigma_n1, gamma_k);

			// Compute the Lambda derivative of plastic strains with respect to the strain that will be used for the AL-method in f_star
			// The last condition checks whether plasti is active at all, if it isn't then Lambda returns as zero (initial value), this just saves some computation time.
			// Using the variable plasti_MatMod would not be sufficient, because this bool is triggered when plasti was triggered the first time and remains true forever.
			// In contrast, we want to know here, whether the current quadrature point is plastic, hence the use of the Lagrange increment
			if ( parameter.solution_method == enums::ArcLength )
			{
				AssertThrow(dim==3, ExcMessage( "MaterialModel::get_StiffnessTensor_C_Elastoplastic:"
												" The elastoplastic code for the arc-length method has not yet been implemented for 2D."
												" Please think about this first and try to adapt the code accordingly."));
				SymmetricTensor<4,3> Lambda_3D = eq_list.get_Lambda_ep(gamma_k);
				Lambda = extract_dim<dim>(Lambda_3D);
			}

			// Update history and stress
			 eps_p_n = eq_list.get_eps_p_n_k_ep(gamma_k);
			 alpha_n = alpha_k;
			 sigma_n1_3D = sigma_n1;

			if ( parameter.write_quad_points == true )
			{
				std::ofstream write_gamma_strain(pre_txt_directory+"gammaStrain.txt", std::ios::app);
				write_gamma_strain << eps_n1_3D[1][1] << " , " << 0 << " , " << 0 << " , " << 0 << " , " << alpha_n << std::endl;
			}
		 }
	}

	// Compute_vM_stress with 3D-stress
	// ToDo-optimize: don't calculate this for every NR-iteration, but just for the converged solution
	 stress_vM = compute_vM_stress ( sigma_n1_3D );
}

//----------------------------------------------------

#endif // MaterialModel_H

