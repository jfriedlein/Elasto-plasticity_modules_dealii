
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

	enum enum_material_parameters
	{
		 kappa = 0,
		 mu = 1,
		 yield_stress = 2,
		 K = 3,
		 yield_stress_incr = 4,
		 K_exp = 5,
		 plastic_hardening = 6
	};
}


/**
 * How the input arguments are mostly designed: \n
 * The stuff that is being updated, hence changes in each iteration usually comes first.
 * Then the *_n values (constant in this Newton-Raphson iteration follow.
 * Thirdly, we get the material parameter (sometimes abbreviated as 'cm') and possibly the Hill tensor.
 * And lastly, if needed the optional damage functions for coupled plasticity-damage
 */
namespace elastoplastic_equations
{
	/**
	 * Verify whether the chosen plastic hardening law exists
	 * @param plastic_hardening
	 */
	void verify_hardening_law ( enums::enum_hardening_law &plastic_hardening )
	{
		AssertThrow( ( 	  plastic_hardening==enums::standard_lin_iso ||
						  plastic_hardening==enums::saturated_alpha ||
						  plastic_hardening==enums::saturated_Voce_hard_stress ||
						  plastic_hardening==enums::saturated_Miehe_hard_stress ),
						ExcMessage( "Elastoplastic_equations<< The selected hardening law could not be verified. "
									"Maybe it is not implemented or you forgot to update the 'verify_hardening_law' "
									"function together with the enumerator 'enum_hardening_law'."));
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

	 // @todo We could such a flag set in the constructor, to raise error messages in case
	 // some functions is called that is only valid for pure plasticity. However, every
	 // AssertThrow cost real money, so we have to see whether that is worth it.
	 //const bool damage_active = false;

	 template<typename Number>
	 Number get_alpha_n_k( const Number &gamma_k, const double &alpha_n, const std::vector<double> &cm )
	 {
		switch ( int(cm[enums::plastic_hardening]) )
		{
			case enums::standard_lin_iso:
				return /*alpha_k =*/  alpha_n + std::sqrt(2./3.) * gamma_k;
			case enums::saturated_alpha:
				return /*alpha_k =*/ (alpha_n + std::sqrt(2./3.) * gamma_k)
									 / (1. + std::sqrt(2./3.) * cm[enums::K]/cm[enums::yield_stress_incr] * gamma_k);
			case enums::saturated_Voce_hard_stress:
				return /*alpha_k =*/  alpha_n + std::sqrt(2./3.) * gamma_k;
			case enums::saturated_Miehe_hard_stress:
				return /*alpha_k =*/  alpha_n + std::sqrt(2./3.) * gamma_k;
//	 		case enums::your_hard_law:
//				return /*alpha_k =*/ ...;
		}
		return false;
	 }
	 //#######################################################################################################################################################
	 template<typename Number>
	 SymmetricTensor<2,3,Number> get_eps_p_n_k( const Number &gamma, const SymmetricTensor<2,3,Number> &n_n1, const SymmetricTensor<2,3,double> &eps_p_n, const Number dmg_p=1. )
	 {
		return eps_p_n + (1. / dmg_p) * gamma * n_n1;
	 }
	 //#######################################################################################################################################################
	 template<typename Number>
	 SymmetricTensor<2,3,Number> get_stress_T_t( const SymmetricTensor<2,3,Number> &eps_n1, const SymmetricTensor<2,3> &eps_p_n, const std::vector<double> &material_parameters )
	 {
		 return /*stress_T_t =*/ material_parameters[enums::kappa] * trace(eps_n1) * unit_symmetric_tensor<3>()
								 + 2. * material_parameters[enums::mu] * ( deviator(eps_n1) - eps_p_n );
	 }
	 //#######################################################################################################################################################
	 template<typename Number>
	 SymmetricTensor<2,3,Number> get_stress_n1( const Number &gamma, const Number &hardStress_R, const SymmetricTensor<2,3,Number> &stress_T_t, const SymmetricTensor<4,3> &HillT_H,
												const std::vector<double> &cm, const Number dmg_mu=1., const Number dmg_p=1. )
	 {
		// start from the trial value
		// Note that for the following to work, namely that we don't use alpha as an input argument, we need to compute the newest alpha
		// via the get_alpha_n_k_ep function (which writes it into the member variable \a alpha_k)
		return invert( identity_tensor<3>() + 2.*cm[enums::mu] * dmg_mu/dmg_p * gamma / ( std::sqrt(2./3.)*(cm[enums::yield_stress] - hardStress_R) ) * HillT_H )
			   * stress_T_t;
//		 return stress_T_t - 2.*cm[enums::mu]*gamma*deviator(stress_T_t)/deviator(stress_T_t).norm();
	 }
	 //#######################################################################################################################################################
	 template<typename Number>
	 SymmetricTensor<4,3,Number> get_Ainv( const Number &gamma, const Number &hardStress_R,
										   const std::vector<double> &cm, const SymmetricTensor<4,3> &HillT_H,
										   const Number dmg_mu=1., const Number dmg_p=1. )
	 {
		return invert( identity_tensor<3>() + 2. * cm[enums::mu] * dmg_mu/dmg_p * gamma / ( std::sqrt(2./3.)*(cm[enums::yield_stress]-hardStress_R)) * HillT_H );
	 }
	 //#######################################################################################################################################################
	 template<typename Number>
	 Number get_d_R_d_gammap( const Number &gamma, const Number &alpha_k, const double &alpha_n, const std::vector<double> &cm )
	 {
		switch ( int(cm[enums::plastic_hardening]) )
		{
			case enums::standard_lin_iso:
				return /*d_R_d_gammap =*/ - cm[enums::K] * std::sqrt(2./3.);
			case enums::saturated_alpha:
				return /*d_R_d_gammap =*/ - cm[enums::K] * ( std::sqrt(2./3.)
														   * (1. - cm[enums::K] / cm[enums::yield_stress_incr] * alpha_n)
														   / std::pow(1. + std::sqrt(2./3.) * cm[enums::K]/cm[enums::yield_stress_incr] * gamma,2) );
			case enums::saturated_Voce_hard_stress:
				return /*d_R_d_gammap =*/ - cm[enums::K] * std::exp( -cm[enums::K]/cm[enums::yield_stress_incr] * alpha_k ) * std::sqrt(2./3.);
			case enums::saturated_Miehe_hard_stress:
				return /*d_R_d_gammap =*/ ( -cm[enums::K] - cm[enums::yield_stress_incr] * cm[enums::K_exp] * std::exp( -cm[enums::K_exp] * alpha_k ))
										  * std::sqrt(2./3.);
//	 		case enums::your_hard_law:
//				return /*d_R_d_gammap =*/ ...;
		}
		return false;
	 }
	 //#######################################################################################################################################################
	 // @todo not compatible to damage, update + Phi_k for all terms
	 template<typename Number>
	 Number get_dPhi_dgamma( const Number &gamma, const Number &hardStress_R, const Number &alpha_k, const Number &Phi_k, const SymmetricTensor<2,3,Number> &n_k,
			 	 	 	     const SymmetricTensor<2,3,Number> &stress_T_t, const double &alpha_n,
							 const std::vector<double> &cm, const SymmetricTensor<4,3> &HillT_H, const Number &dmg_mu=1., const Number &dmg_p=1. )
	 {
		SymmetricTensor<4,3> A_inv = get_Ainv( gamma, hardStress_R, cm, HillT_H, dmg_mu, dmg_p );
		// The use of the newest yield function seems to be quite quite advantages (reduces nbr of qp iterations by one,
		// and for linear isotropic hardening instead of five iterations, we get the desired one-step solution)
		return - n_k  * (
							(A_inv*A_inv)
							* HillT_H
							* 2. * cm[enums::mu] / ( std::sqrt(2./3.) * (cm[enums::yield_stress]-hardStress_R) + Phi_k )
							* ( 1. + gamma / ( std::sqrt(1.5)*Phi_k + (cm[enums::yield_stress]-hardStress_R) ) * get_d_R_d_gammap(gamma, alpha_k, alpha_n, cm) )
						)
					  * stress_T_t
			   + std::sqrt(2./3.) * get_d_R_d_gammap(gamma, alpha_k, alpha_n, cm);

//		return -( 2*cm[enums::mu] + 2./3. * cm[enums::K] );
	 }
	 //#######################################################################################################################################################
	 template<typename Number>
	 Number get_hardeningStress_R( const Number &alpha_k, const std::vector<double> &cm )
	 {
		switch ( int(cm[enums::plastic_hardening]) )
		{
			case enums::standard_lin_iso:
				return /*hardStress_R =*/ - cm[enums::K] * alpha_k;
			case enums::saturated_alpha:
				return /*hardStress_R =*/ - cm[enums::K] * alpha_k;
			case enums::saturated_Voce_hard_stress:
				return /*hardStress_R =*/ - cm[enums::yield_stress_incr] * ( 1. - std::exp(-cm[enums::K] / cm[enums::yield_stress_incr] * alpha_k) );
			case enums::saturated_Miehe_hard_stress:
				return /*hardStress_R =*/ - cm[enums::K] * alpha_k - cm[enums::yield_stress_incr] * ( 1 - std::exp(-cm[enums::K_exp] * alpha_k) );
//	 		case enums::your_hard_law:
//				return /*hardStress_R =*/ ...;
		}
		return false;
	 }
	 //#######################################################################################################################################################
	 template<typename Number>
	 Number get_yielding_norm( const SymmetricTensor<2,3,Number> &tensor, const SymmetricTensor<4,3> &HillT_H, const bool &GG_mode_active=false, bool &GG_mode_requested=false )
	 {
		// @todo-ensure: How can the Hill-norm get negative?
		double tmp = tensor * HillT_H * tensor;
		if ( tmp<0 )
		{
			// If the GG-mode is active, then we request a restart (=true) and overwrite the negative value by just something.
			if ( GG_mode_active )
			{
				GG_mode_requested=true;
				std::cout << "get_yielding_norm<< The Hill norm of the stress tensor got negative as " << tmp << "." << std::endl;
				tmp = 9e9;
			}
			else
				AssertThrow( false, ExcMessage("elpl_equation_list<< The Hill norm of the stress tensor got negative as "+std::to_string(tmp)+"."));
		}
		return std::sqrt(tmp);
	 }
	 //#######################################################################################################################################################
	 template<typename Number>
	 SymmetricTensor<2,3,Number> get_n_n1( SymmetricTensor<2,3,Number> &stress_k, const SymmetricTensor<4,3> &HillT_H, const bool &GG_mode_active=false, bool &GG_mode_requested=false )
	 {
		// For zero strain (initial) we get zero stress and 0/0 doesn't work
		 Number denominator = (stress_k.norm()==0.) ? 1. : get_yielding_norm( stress_k, HillT_H, GG_mode_active, GG_mode_requested );
		return (HillT_H * stress_k) / denominator;
	 }
	 //#######################################################################################################################################################
	 // @todo We run into trouble here with our optional arguments, I would like to be able to use GG-mode options also without the damage option
	 template<typename Number>
	 Number get_plastic_yield_fnc ( const SymmetricTensor<2,3,Number> &stress_k, const Number &hardStress_R,
									const std::vector<double> &cm, const SymmetricTensor<4,3> &HillT_H,
									const double &dmg_p=1., const bool &GG_mode_active=false, bool &GG_mode_requested=false )
	 {
		// Note that for the following to work, namely that we don't use alpha as an input argument, we need to compute the newest alpha
		// via the get_alpha_n_k_ep function (which writes it into the member variable \a alpha_k)
		 return 1./dmg_p * get_yielding_norm( stress_k, HillT_H, GG_mode_active, GG_mode_requested ) - std::sqrt(2./3.) * ( cm[enums::yield_stress] - hardStress_R );
	 }
	 //#######################################################################################################################################################
	 // The following is only used for analytical tangents (because it's the analytical tangent), hence we need to \a Number template
	 //#######################################################################################################################################################
//	 template<typename Number>
//	 SymmetricTensor<4,3> get_Lambda_ep( const Number &gamma_input )
//	 {
//		AssertThrow( false, ExcMessage("elpl_equation_list<< get_Lambda_ep is untested, equation just fills some space"));
//
//	 	return - 2.*mu / -(get_dPhi_dgamma_ep(alpha_n,n_n1)) * outer_product(n_n1,n_n1)
//			   - gamma / stress_T_t_norm
//				 * ( HillT_H * d_Tt_d_eps
//				      - 1. / stress_T_t_norm
//					    * outer_product( n_n1, ( (HillT_H * stress_T_t) * d_Tt_d_eps ) )
//				   );
//	 }
	 //#######################################################################################################################################################
	 SymmetricTensor<4,3> get_N_four( const SymmetricTensor<2,3> &stress_n1, const SymmetricTensor<4,3> &HillT_H )
	 {
		 return std::pow( stress_n1 * HillT_H * stress_n1, -1.5 )
				* ( HillT_H * ( stress_n1 * HillT_H * stress_n1 )
					- outer_product(HillT_H*stress_n1, stress_n1*HillT_H) );
	 }
	 //#######################################################################################################################################################
	 /**
	  * @note Only valid for pure plasticity (not for damage)
	  */
	 SymmetricTensor<4,3> get_tangent_plastic( const SymmetricTensor<2,3> &stress_n1, const double &gamma, const SymmetricTensor<2,3> &n_n1,
			 	 	 	 	 	 	 	 	   const double &alpha_k, const double &alpha_n,
											   const std::vector<double> &cm, const SymmetricTensor<4,3> &HillT_H )
	 {
		SymmetricTensor<4,3> N_four = get_N_four( stress_n1, HillT_H );
		SymmetricTensor<4,3> d_Tt_d_eps = cm[enums::kappa] * outer_product( unit_symmetric_tensor<3>(), unit_symmetric_tensor<3>() )
										  + 2. * cm[enums::mu] * deviator_tensor<3>();
		SymmetricTensor<4,3> E_e = invert( invert(d_Tt_d_eps) + gamma * N_four );

		return E_e - 1. / ( n_n1 * E_e * n_n1 - std::sqrt(2./3.) * get_d_R_d_gammap(gamma, alpha_k, alpha_n, cm) ) * ( outer_product(E_e*n_n1, n_n1*E_e) );

//		 return cm[enums::kappa] * StandardTensors::IxI<3>()
//				+ 2. * cm[enums::mu] * StandardTensors::I_deviatoric<3>()
//				- 4.*cm[enums::mu]*cm[enums::mu] / ( 2.*cm[enums::mu] + 2./3. * cm[enums::K] ) * outer_product(n_n1,n_n1)
//				- 4.*cm[enums::mu]*cm[enums::mu] * gamma / deviator(stress_T_t).norm() * ( StandardTensors::I_deviatoric<3>() - outer_product(n_n1,n_n1) );
	 }
 }


