
#include <deal.II/lac/vector.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>

//#include "auxiliary_functions/StandardTensors.h"
//
#include "../MA-Code/enumerator_list.h"

using namespace dealii;

// Addition to the enumerator list
namespace enums
{
	enum enum_hardening_law
	{
		 standard_lin_iso = 0,
		 saturated_alpha = 1,
		 saturated_Voce_hard_stress = 2,
		 saturated_Miehe_hard_stress = 3
	};
}


SymmetricTensor<4,3> set_Hill_tensor( const std::vector<double> &Hill_coefficients, const double &sheet_orientation_theta )
{
	 // set up the hill tensor based on the hill coefficients
	 // Following the paper "Anisotropic additive plasticity in the logarithmic strain space"
	 // by Miehe et al. eq. (3.40)-(3.46)
	 // @note In this section we partly work around the "issue" "C++ starts counting at 0"
	 std::vector<double> alpha_ (1+9); // we ignore the first entry
	 alpha_[1] = 2./3. * std::pow(Hill_coefficients[enums::h11],-2);
	 alpha_[2] = 2./3. * std::pow(Hill_coefficients[enums::h22],-2);
	 alpha_[3] = 2./3. * std::pow(Hill_coefficients[enums::h33],-2);
	 alpha_[7] = 1./3. * std::pow(Hill_coefficients[enums::h12],-2);
	 alpha_[8] = 1./3. * std::pow(Hill_coefficients[enums::h23],-2);
	 alpha_[9] = 1./3. * std::pow(Hill_coefficients[enums::h31],-2);
	 alpha_[4] = 0.5 * ( alpha_[3] - alpha_[1] - alpha_[2] );
	 alpha_[5] = 0.5 * ( alpha_[1] - alpha_[2] - alpha_[3] );
	 alpha_[6] = 0.5 * ( alpha_[2] - alpha_[1] - alpha_[3] );

	 // The Hill tensor in matrix representation
	 FullMatrix<double> H_matrix(6,6);
	 {
		 H_matrix(0,0) = alpha_[1];
		 H_matrix(1,1) = alpha_[2];
		 H_matrix(2,2) = alpha_[3];
		 H_matrix(3,3) = 0.5 * alpha_[7];
		 H_matrix(4,4) = 0.5 * alpha_[8];
		 H_matrix(5,5) = 0.5 * alpha_[9];

		 H_matrix(0,1) = alpha_[4];
		 H_matrix(1,0) = H_matrix(0,1);

		 H_matrix(1,2) = alpha_[5];
		 H_matrix(2,1) = H_matrix(1,2);

		 H_matrix(0,2) = alpha_[6];
		 H_matrix(2,0) = H_matrix(0,2);
	 }

	 // orthogonal basis by three orthogonal directions a_i
	 std::vector< Tensor<1,3> > a_ (3);
	 {
		 // first basis vector (for theta=0° equal to x-axis)
		  a_[0][enums::x] = std::cos( sheet_orientation_theta/180. * 4. * std::atan(1) );
		  a_[0][enums::y] = std::sin( sheet_orientation_theta/180. * 4. * std::atan(1) );
		  a_[0][enums::z] = 0;
		 // second basis vector (for theta=0° equal to y-axis)
		  a_[1][enums::x] = - std::sin( sheet_orientation_theta/180. * 4. * std::atan(1) );
		  a_[1][enums::y] =   std::cos( sheet_orientation_theta/180. * 4. * std::atan(1) );
		  a_[1][enums::z] = 0;
		 // third basis vector (for sheets always along z-axis)
		  a_[2][enums::z] = 1.; // x and y components of this vector are default zero
	 }

	 // @todo: are the tensors in \a m symmetric then?, use _sym function
	 std::vector< std::vector< SymmetricTensor<2,3> > > m_ (1+3,std::vector< SymmetricTensor<2,3> >(1+3));
	 {
		 for ( unsigned int i=0; i<3; i++ )
			 for ( unsigned int j=0; j<3; j++ )
				 m_[i+1][j+1] = 0.5 * symmetrize( outer_product( a_[i], a_[j] ) + outer_product( a_[j], a_[i] ) );
	 }

	 // @todo-optimize: Use the outer_product_sym functions
	 // @todo: What about the goofy factor of 2 for isotropic???
	 SymmetricTensor<4,3> HillT_H;
	   HillT_H = alpha_[1] * outer_product( m_[1][1], m_[1][1] )
			   + alpha_[2] * outer_product( m_[2][2], m_[2][2] )
			   + alpha_[3] * outer_product( m_[3][3], m_[3][3] )
			   + alpha_[4] * 0.5 * ( outer_product( m_[1][1], m_[2][2] ) + outer_product( m_[2][2], m_[1][1] ) ) * 2. /*factor of 2?????*/
			   + alpha_[5] * 0.5 * ( outer_product( m_[2][2], m_[3][3] ) + outer_product( m_[3][3], m_[2][2] ) ) * 2. /*factor of 2?????*/
			   + alpha_[6] * 0.5 * ( outer_product( m_[1][1], m_[3][3] ) + outer_product( m_[3][3], m_[1][1] ) ) * 2. /*factor of 2?????*/
			   + alpha_[7] * 2. * outer_product( m_[1][2], m_[2][1] )
			   + alpha_[8] * 2. * outer_product( m_[2][3], m_[3][2] )
			   + alpha_[9] * 2. * outer_product( m_[1][3], m_[3][1] );

	if ( ((HillT_H * StandardTensors::I<3>()).norm() + (StandardTensors::I<3>() * HillT_H).norm()) > 1e-14 )
	{
		std::cout << std::endl;
		std::cout<< std::scientific;
		std::cout << "HillT_H<< Hill Tensor not purely deviatoric, wrong setup of the equations. Results in "
				  << (HillT_H * StandardTensors::I<3>()).norm()
				  << " instead of less than 1e-14 (numercially zero)." << std::endl;
		AssertThrow(false, ExcMessage("HillT_H<< Hill Tensor not purely deviatoric"));
	}
//		 AssertThrow( ( (HillT_H * StandardTensors::I<3>()).norm() + (StandardTensors::I<3>() * HillT_H).norm()) < 1e-20,
//				      ExcMessage("HillT_H<< Hill Tensor not purely deviatoric, wrong setup of the equations. Results in "
//				    		     +std::to_string((HillT_H * StandardTensors::I<3>()).norm())+" instead of less than 1e-20 (numercially zero)."));
	return HillT_H;
}


template<int dim, typename Number>
class elastoplastic_equations
 {
 public:
	elastoplastic_equations( const unsigned int plastic_hardening, const double &mu, const double &kappa, const double &yield_stress,
							 const double &K, const double &yield_stress_incr, const double &K_exp,
							 const SymmetricTensor<2,3> &eps_n1, const SymmetricTensor<2,3> &eps_p_n, const double &alpha_n, const SymmetricTensor<4,3> &HillT_H )
	:
	plastic_hardening(plastic_hardening),
	mu(mu),
	kappa(kappa),
	yield_stress(yield_stress),
	K(K),
	yield_stress_incr(yield_stress_incr),
	K_exp(K_exp),
	eps_n1(eps_n1),
	eps_p_n(eps_p_n),
	alpha_n(alpha_n),
	HillT_H(HillT_H)
	{
		stress_T_t = kappa * trace(eps_n1) * unit_symmetric_tensor<3>()
					 + 2. * mu * ( deviator(eps_n1) - eps_p_n );
		d_Tt_d_eps = kappa * outer_product( unit_symmetric_tensor<3>(), unit_symmetric_tensor<3>() )
					 + 2. * mu * deviator_tensor<3>();
		// The Hill tensor was already set in the parameter file and possibly updated for anisotropy
		// in the main function, so we can simply use it here containing its final values
		 n_n1 = (HillT_H * stress_T_t) / std::sqrt( stress_T_t * HillT_H * stress_T_t );
	}

 //private:
 public:
	const unsigned int plastic_hardening;
	
	 enum enum_get_elpl
	 {
		 get_alpha = 0,
		 get_R = 1,
		 get_dR_dg = 2
		 
	 };

	 // ToDo:
	 // check how to implement/use the parameter like K, yield_stress_incr;
	 // maybe into constructor or use full parameter argument
	 // or use a std::vector that contains all the values and is
	 // accessed only via enumerators to identify which entry corresponds
	 // to which variableD
	 const double mu;
	 const double kappa;
	 const double yield_stress;
	 const double K;
	 const double yield_stress_incr;
	 const double K_exp;
	 
	 SymmetricTensor<4,3> HillT_H;

	 // ToDo-optimize: Think about using pointers, etc. to avoid the local copies of 
	 // these variables in the normal code and in this class. Or store the values
	 // only in this class and access them through this class.
	 const SymmetricTensor<2,3,Number> eps_n1;
	 const Sacado_Wrapper::SymTensor<3> eps_n1_Sac;
	 const SymmetricTensor<2,3> eps_p_n;

	 SymmetricTensor<2,3,Number> stress_T_t;
	 SymmetricTensor<2,3,Number> n_n1;

	 Number alpha_k, gamma, R;
	 double alpha_n;
	 Number d_R_d_gamma;
	 Number d_Phi_d_gamma;

	 SymmetricTensor<4,3> d_Tt_d_eps;
	 
 public:
	// Reset the Hill tensor in case it shall not be the deviatoric tensor
	// @todo-optimize: Maybe add an overloaded variant that enables the input
	// of the classical 6x6 matrix for H.
	 
   // Accessor functions: \n
   // 1. Update the member variables with the new values from the input arguments \n
   // 2. Call the equation list function and evaluate the desired entry \n
   // 3. Return the updated member variable
	 Number get_alpha_n_k_ep( const Number &gamma_input )
	 {
		gamma = gamma_input;
		elastoplastic_equations_list ( get_alpha );

		return alpha_k;
	 };
	 //#######################################################################################################################################################
	 SymmetricTensor<2,3,Number> get_eps_p_n_k_ep( const Number &gamma_input )
	 {
		gamma = gamma_input;
		return eps_p_n + gamma * n_n1;
	 };
	 //#######################################################################################################################################################
	 SymmetricTensor<2,3,Number> get_stress_T_t_ep(  )
	 {
		return stress_T_t;
	 };
	 //#######################################################################################################################################################
	 SymmetricTensor<2,3,Number> get_stress_n1_ep( const Number &gamma_input, const Number &alpha_k_input )
	 {
		gamma = gamma_input;
		// start from the trial value not the last \a sigma_n1
		// where the update of the entire stress tensor equals the update of the deviatoric stress part
		//return stress_T_t - 2. * mu * gamma * n_n1;
		return invert( StandardTensors::II<3>() + 2.*mu*gamma/(std::sqrt(2./3.)*(yield_stress-get_hardeningStress_R_ep( alpha_k_input )) ) * HillT_H ) * stress_T_t;
//		return stress_T_t - 2. * mu * gamma * n_n1;
//		return stress_km1 - 2. * mu * gamma_update * n_n1;
	 };
	 //#######################################################################################################################################################
//	 Number get_dPhi_dgamma_ep( const double &alpha_n_input, SymmetricTensor<2,3,Number> &sigma_n1 )
//	 {
//	 	alpha_n = alpha_n_input;
//		SymmetricTensor<4,3> N_four = std::pow( sigma_n1 * HillT_H * sigma_n1, -1.5 ) * ( HillT_H * ( sigma_n1 * HillT_H * sigma_n1 ) - outer_product(HillT_H*sigma_n1, sigma_n1*HillT_H) );
//
//	 	AssertThrow( false, ExcMessage("elpl_equation_list<< dPhi_dgamma is OoO"));
//
//	 	d_Phi_d_gamma = - 2. * mu * n_n1 * n_n1
//	 					//- 2. * mu * gamma * n_n1 * N_four * (-2. * mu * n_n1)
//	 			        + std::sqrt( 2./3. ) * get_d_R_d_gamma_ep(alpha_n);
//
//	 	return d_Phi_d_gamma;
//	 };
	 //#######################################################################################################################################################
	 Number get_dPhi_dgamma_ep( const double &gamma_k )
	 {
		double R = get_hardeningStress_R_ep( get_alpha_n_k_ep(gamma_k) );
		SymmetricTensor<4,3> A_inv = invert( StandardTensors::II<3>() + 2.*mu*gamma_k / ( std::sqrt(2./3.)*(yield_stress-R) ) * HillT_H );

//		return - 1. / ( 2. * (Phi_k + sqrt2_3 * (yield_stress-R))*(sqrt2_3 * (yield_stress-R)) )
//				 * ( eq_list.stress_T_t *
//											 (eq_list.HillT_H * eq_list.d_Tt_d_eps * ( A_inv * A_inv ) * eq_list.HillT_H * A_inv
//											 + A_inv * eq_list.HillT_H * ( A_inv * A_inv ) * eq_list.d_Tt_d_eps * eq_list.HillT_H )
//					 * eq_list.stress_T_t )
//			   + std::sqrt( 2./3. ) * get_d_R_d_gamma_ep(alpha_n);

		// @todo: The following only achieves superlinear behaviour close to the solution, before it's quadratic
		// The shorter one works very well for isotropy and better than the second for aniso
		return -2. * mu * n_n1 * A_inv * n_n1 + std::sqrt( 2./3. ) * get_d_R_d_gamma_ep() ;
		//return n_n1 * (-2.*mu / (std::sqrt( 2./3. ) * (yield_stress-R)) * (A_inv*A_inv) * HillT_H * stress_T_t) + std::sqrt( 2./3. ) * get_d_R_d_gamma_ep();
	 }
	 //#######################################################################################################################################################
	 Number get_hardeningStress_R_ep( const Number &alpha_k_input )
	 {
	 	alpha_k = alpha_k_input;
	 	elastoplastic_equations_list ( get_R );
	 	return R;
	 };
	 //#######################################################################################################################################################
	 Number get_d_R_d_gamma_ep( )
	 {
	 	elastoplastic_equations_list ( get_dR_dg );
	 	return d_R_d_gamma;
	 };
	 //#######################################################################################################################################################
	 SymmetricTensor<2,3,Number> get_n_n1_ep( SymmetricTensor<2,3,Number> &stress_k )
	 {
		update_n_n1_ep(stress_k);
	 	return n_n1;
	 };
	 //#######################################################################################################################################################
	 void update_n_n1_ep( const SymmetricTensor<2,3,Number> &stress_k )
	 {
		n_n1 = (HillT_H * stress_k) / get_yielding_norm( stress_k );
	 };
	 //#######################################################################################################################################################
	 Number get_yielding_norm( const SymmetricTensor<2,3,Number> &tensor )
	 {
		// @todo-ensure: How can the Hill-norm get negative?
		double tmp = tensor * HillT_H * tensor;
		AssertThrow( tmp>=0, ExcMessage("elpl_equation_list<< The Hill norm of the stress tensor got negative."));
		return std::sqrt(tmp);
		//return std::sqrt( tensor * HillT_H * tensor );
	 };
	 //#######################################################################################################################################################
	 Number get_plastic_yield_fnc ( const SymmetricTensor<2,3,Number> &stress_k, Number &alpha_k_input )
	 {
		 return get_yielding_norm( stress_k ) - std::sqrt(2./3.) * ( yield_stress - get_hardeningStress_R_ep( alpha_k_input ) );
	 }
	 //#######################################################################################################################################################
	 SymmetricTensor<4,3> get_Lambda_ep( const Number &gamma_input )
	 {
		gamma = gamma_input;
		
		double stress_T_t_norm = get_yielding_norm( stress_T_t );

	 	AssertThrow( false, ExcMessage("elpl_equation_list<< get_Lambda_ep is untested, equation just fills some space"));
	 	
//	 	return - 2.*mu / -(get_dPhi_dgamma_ep(alpha_n,n_n1)) * outer_product(n_n1,n_n1)
//			   - gamma / stress_T_t_norm
//				 * ( HillT_H * d_Tt_d_eps
//				      - 1. / stress_T_t_norm
//					    * outer_product( n_n1, ( (HillT_H * stress_T_t) * d_Tt_d_eps ) )
//				   );
	 };
	 //#######################################################################################################################################################
	 SymmetricTensor<4,3> get_tangent_plastic_ep( const SymmetricTensor<2,3> &sigma_n1, const Number &gamma_k )
	 {
		SymmetricTensor<4,3> N_four = std::pow( sigma_n1 * HillT_H * sigma_n1, -1.5 ) * ( HillT_H * ( sigma_n1 * HillT_H * sigma_n1 ) - outer_product(HillT_H*sigma_n1, sigma_n1*HillT_H) );
		SymmetricTensor<4,3> E_e = invert( invert(d_Tt_d_eps) + gamma_k * N_four );

		return E_e - 1. / ( n_n1 * E_e * n_n1 - std::sqrt( 2./3. ) * get_d_R_d_gamma_ep() ) * ( outer_product(E_e*n_n1, n_n1*E_e) );
	 };
	 
   // Summary of equations for the different hardening types
	 void elastoplastic_equations_list( enum_get_elpl get_elpl )
	 {
	 	switch ( plastic_hardening )
	 	{
			case enums::standard_lin_iso: { switch ( get_elpl ) {
				//############################################################################################################################################
 				case get_alpha:/*#######*/ alpha_k = alpha_n + std::sqrt(2./3.) * gamma;  /*#########################################################*/ break;
				//############################################################################################################################################
				case get_R:/*###########*/ R = - K * alpha_k; /*#####################################################################################*/ break;
				//############################################################################################################################################
				case get_dR_dg:/*#######*/ d_R_d_gamma = - K * std::sqrt(2./3.); /*##################################################################*/ break;
				//############################################################################################################################################
				}
			}
			break;
	 		case enums::saturated_alpha: { switch ( get_elpl ) {
	 			//############################################################################################################################################
				case get_alpha:/*#######*/ alpha_k = (alpha_n + std::sqrt(2./3.) * gamma) / (1. + std::sqrt(2./3.) * K/yield_stress_incr * gamma); /*#*/break;
				//############################################################################################################################################
				case get_R:/*###########*/ R = - K * alpha_k; /*#####################################################################################*/ break;
				//############################################################################################################################################
				case get_dR_dg:/*#######*/ d_R_d_gamma = - K * ( std::sqrt(2./3.)
																 * (1. - K/yield_stress_incr * alpha_n)
						 	 	 	 	 	 	 	 	 	 	 / std::pow(1. + std::sqrt(2./3.) * K/yield_stress_incr * gamma,2) ); /*#############*/ break;
				//############################################################################################################################################
				}
	 		}
	 		break;
	 		case enums::saturated_Voce_hard_stress: { switch ( get_elpl ) {
	 			//############################################################################################################################################
				case get_alpha:/*#######*/ alpha_k = alpha_n + std::sqrt(2./3.) * gamma; /*##########################################################*/ break;
				//############################################################################################################################################
				case get_R:/*###########*/ R = - yield_stress_incr * ( 1 - std::exp(-K/yield_stress_incr * alpha_k) ); /*############################*/ break;
				//############################################################################################################################################
				case get_dR_dg:/*#######*/ d_R_d_gamma = - K * std::exp( -K/yield_stress_incr * alpha_k ) * std::sqrt(2./3.); /*#####################*/ break;
				//############################################################################################################################################
				}
	 		}
	 		break;
	 		case enums::saturated_Miehe_hard_stress: { switch ( get_elpl ) {
	 			//############################################################################################################################################
				case get_alpha:/*#######*/ alpha_k = alpha_n + std::sqrt(2./3.) * gamma; /*##########################################################*/ break;
				//############################################################################################################################################
				case get_R:/*###########*/ R = - K * alpha_k - yield_stress_incr * ( 1 - std::exp(-K_exp * alpha_k) ); /*############################*/ break;
				//############################################################################################################################################
				case get_dR_dg:/*#######*/ d_R_d_gamma = (- K - yield_stress_incr * K_exp * std::exp( -K_exp * alpha_k )) * std::sqrt(2./3.); /*#####*/ break;
				//############################################################################################################################################
				}
	 		}
	 		break;
	 		// In essence, we take all this effort to enable us to implementen different hardening laws simply by the set of three equations
	 		// * evolution equation for the internal variable \a alpha_k
	 		// * hardening stress \a R
	 		// * derivative of the \a R wrt to the plastic Lagrange multiplier \a gamma
	 		// After you derived and implemented these three equations for your hardening law, everything else is handled by this class. So, you can
	 		// iterate on the qp level by using \a d_Phi_d_gamma to update the Lagrange multiplier and use the tangent simply by calling the respective function.
//	 		case enums::your_hard_law: { switch ( get_elpl ) {
//	 			//############################################################################################################################################
//				case get_alpha:/*#######*/ alpha_k = ...; /*##########################################################*/ break;
//				//############################################################################################################################################
//				case get_R:/*###########*/ R = ...; /*############################*/ break;
//				//############################################################################################################################################
//				case get_dR_dg:/*#######*/ d_R_d_gamma = ...; /*#####*/ break;
//				//############################################################################################################################################
//				}
//	 		}
//	 		break;
 		}
	 };
 };
