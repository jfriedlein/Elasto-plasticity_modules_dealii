
#include <deal.II/lac/vector.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>

//#include "auxiliary_functions/StandardTensors.h"
//
#include "../MA-Code/enumerator_list.h"

// for get_value(*)
#include "../Sacado_Tensors_dealII/Sacado_Wrapper.h"

using namespace dealii;

// Addition to the enumerator list
namespace enums
{
	enum enum_hardening_law
	{
		 standard_lin_iso = 0,
		 saturated_alpha = 1,
		 saturated_Voce_hard_stress = 2,
		 saturated_Miehe_hard_stress = 3,
		 exponent_exponential_hard_stress = 4,
		 K_alpha_exp = 5
	};

	enum enum_P_kinHard_law
	{
		 P_kinHard_standard_lin = 0,
		 P_kinHard_saturated = 1
	};

	enum enum_material_parameters
	{
		 kappa = 0,
		 mu = 1,
		 yield_stress = 2,
		 K = 3,
		 yield_stress_incr = 4,
		 K_exp = 5,
		 plastic_hardening = 6,
		 plastic_aniso = 7,
		 P_hard_iso_kin = 8,
		 kin_hard_mod = 9,
		 n_entries = 10
	};
}


/**
 * How the input arguments are mostly designed: \n
 * The stuff that is being updated, hence changes in each iteration usually comes first.
 * Then the *_n values (constant in this Newton-Raphson iteration follow.
 * Thirdly, we get the material parameter (sometimes abbreviated as 'cm') and possibly the Hill tensor.
 * And lastly, if needed the optional damage functions for coupled plasticity-damage
 * @note The functions are arranged in such a way that the definitions are put before the usage.
 */
namespace elastoplastic_equations
{
	/**
	 * Verifies whether the chosen plastic hardening law exists
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

		 // The \a H_matrix seems to only work for the global coordinate system.
		 // For rotated ones, we seem to need the following.

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
			case enums::exponent_exponential_hard_stress:
				return /*alpha_k =*/  alpha_n + std::sqrt(2./3.) * gamma_k;
			case enums::K_alpha_exp:
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
		return SymmetricTensor<2,3,Number>(eps_p_n) + (1. / dmg_p) * gamma * n_n1;
	 }
	 //#######################################################################################################################################################
	 template<typename Number>
	 SymmetricTensor<2,3,Number> get_stress_T_t( const SymmetricTensor<2,3,Number> &eps_n1, const SymmetricTensor<2,3> &eps_p_n,
			 	 	 	 	 	 	 	 	 	 const std::vector<double> &material_parameters, const Number dmg_kappa=1., const Number dmg_mu=1. )
	 {
		 return /*stress_T_t =*/ dmg_kappa * material_parameters[enums::kappa] * trace(eps_n1) * unit_symmetric_tensor<3,Number>()
								 + 2. * dmg_mu * material_parameters[enums::mu] * ( deviator(eps_n1) - eps_p_n );
	 }
	 //#######################################################################################################################################################
	 /**
	  * That presumably only works for the driving force that has been used for the yield function
	  * Computing the inverse of the fourth order tensor \a A, optionally we can incorporate the
	  * yield function \a Phi_k.
	  * @todo However, it converges better if the leave \a Phi_k as zero.
	  * @todo Kinematic hardening is only implemented as a special case and not yet modular.
	  * @param gamma
	  * @param hardStress_R
	  * @param Phi_k Not called by reference to be able to call get_Ainv(..., Number(0), ...) instead of
	  * having to declare a variable for this each time the function is called.
	  * @param cm
	  * @param HillT_H
	  * @param dmg_mu
	  * @param dmg_p
	  * @return
	  */
	 template<typename Number>
	 SymmetricTensor<4,3,Number> get_Ainv( const Number &gamma, const Number &hardStress_R, const Number Phi_k,
										   const std::vector<double> &cm, const SymmetricTensor<4,3> &HillT_H,
										   const Number dmg_mu=1., const Number dmg_p=1., const bool stressRel_Flag=false )
	 {
		return invert<3,Number>( identity_tensor<3,Number>()
								 + (
										 2. * cm[enums::mu] * dmg_mu/dmg_p
										 + cm[enums::kin_hard_mod] / dmg_p * double(stressRel_Flag)
								   ) * gamma / ( std::sqrt(2./3.)*(cm[enums::yield_stress]-hardStress_R) + Phi_k )
								   * SymmetricTensor<4,3,Number>(HillT_H) );
	 }
	 //#######################################################################################################################################################
	 // @note We use \a HillT_H as constant input argument, but transform it to a SymmetricTensor<4,3,Number>, so all the tensor operations work
	 template<typename Number>
	 SymmetricTensor<2,3,Number> get_stressRel_Xsi_n1( const Number &gamma, const Number &hardStress_R, const SymmetricTensor<2,3,Number> &stressRel_Xsi_t,
			 	 	 	 	 	 	 	 	 		   const SymmetricTensor<4,3> &HillT_H, const std::vector<double> &cm,
													   const Number dmg_mu=1., const Number dmg_p=1., const bool stressRel_Flag=true )
	 {
		// start from the trial value
//		return invert<3,Number>( identity_tensor<3,Number>()
//								 + 2.*cm[enums::mu] * dmg_mu/dmg_p * gamma / ( std::sqrt(2./3.)*(cm[enums::yield_stress] - hardStress_R) ) * SymmetricTensor<4,3,Number>(HillT_H) )
//			   * stress_T_t;

		 if ( cm[enums::plastic_aniso] > 0.5 ) // plastic anisotropy
			 return get_Ainv( gamma, hardStress_R, /*Phi_k=*/Number(0), cm, HillT_H, dmg_mu, dmg_p, stressRel_Flag ) * stressRel_Xsi_t;
		 else // isotropic
		 {
			 SymmetricTensor<2,3,Number> n_n1;
			 // If the stress is zero, then the evolution would not be defined (nan).
			 // So we catch this case and only compute \a n_n1 for nonzero stress.
			 // Else we leave the zero entries in \a n_n1 from the declaration.
			  if ( deviator<3,Number>(stressRel_Xsi_t).norm()!=0 )
				 n_n1 = deviator<3,Number>(stressRel_Xsi_t) / deviator<3,Number>(stressRel_Xsi_t).norm();
			 return stressRel_Xsi_t - ( 2. * cm[enums::mu] * dmg_mu + stressRel_Flag * cm[enums::kin_hard_mod] ) * gamma/dmg_p * n_n1 ;
		 }
	 }
	 //#######################################################################################################################################################
	 /**
	  * @todo Study this in more detail: This function exists parallel to get_stressRel_Xsi_n1,
	  * because calling the latter to get the true stress even when switching the
	  * kinematic hardening contribution off, does not work.
	  * @todo Could alternative use Xsi and subtract the backstress_k
	  * @param gamma
	  * @param hardStress_R
	  * @param stress_T_t
	  * @param n_k The NEWEST evolution direction
	  * @param HillT_H
	  * @param cm
	  * @param dmg_mu
	  * @param dmg_p
	  * @return
	  */
	 template<typename Number>
	 SymmetricTensor<2,3,Number> get_stress_n1( const Number &gamma, const SymmetricTensor<2,3,Number> &stress_T_t, const SymmetricTensor<2,3,Number> &n_k,
			 	 	 	 	 	 	 	 	 	const std::vector<double> &cm, const Number dmg_mu=Number(1.), const Number dmg_p=Number(1.) )
	 {
		 // We require the input argument \a n_k to be the newest evolution direction.
		 // Only then the following stress update is also valid for anisotropic plasticity.
		 return stress_T_t - 2. * cm[enums::mu] * dmg_mu/dmg_p * gamma * n_k ;
	 }
	 // Trying to improve the AD convergence by differentiating between +Phi_k terms and +Phi_n1=0
//	 SymmetricTensor<2,3,fad_double> get_stress_n1( const fad_double &gamma, const fad_double &hardStress_R, const SymmetricTensor<2,3,fad_double> &stress_T_t, const double &Phi_k,
//			 	 	 	 	 	 	 	 	 	const SymmetricTensor<4,3> &HillT_H, const std::vector<double> &cm, const fad_double dmg_mu=1., const fad_double dmg_p=1. )
//	 {
//		// start from the trial value
//		 SymmetricTensor<2,3,fad_double> stress_n1 = invert<3,fad_double>( identity_tensor<3,fad_double>()
//								 + 2.*cm[enums::mu] * dmg_mu/dmg_p * gamma / ( std::sqrt(2./3.)*(cm[enums::yield_stress] - hardStress_R) + Phi_k ) * SymmetricTensor<4,3,fad_double>(HillT_H) )
//			   * stress_T_t;
//		 // Now we compute the value without Phi_k
//		 SymmetricTensor<2,3,fad_double> stress_n1_old = invert<3,fad_double>( identity_tensor<3,fad_double>()
//								 + 2.*cm[enums::mu] * dmg_mu/dmg_p * gamma / ( std::sqrt(2./3.)*(cm[enums::yield_stress] - hardStress_R) ) * SymmetricTensor<4,3,fad_double>(HillT_H) )
//			   * stress_T_t;
//		 std::cout << "stres_n1_old1 " << stress_n1_old << std::endl;
//		 // The only thing we want to keep from the computation with Phi_k is the derivative wrt to gamma_k
//			for ( unsigned int i=0; i<3; i++ )
//				for ( unsigned int j=0; j<3; j++ )
//				{
//					double *derivs = &stress_n1[i][j].fastAccessDx(0);
//					double *derivs_old = &stress_n1_old[i][j].fastAccessDx(0);
//
//					derivs_old[6] = derivs[6];
//				}
//			 std::cout << "stres_n1_old2 " << stress_n1_old << std::endl;
//
//		 return stress_n1_old;
//	 }
	 //#######################################################################################################################################################
	 template<typename Number>
	 Number get_d_R_d_gammap( const Number &gamma, const Number &alpha_k, const double &alpha_n, const std::vector<double> &cm )
	 {
		// d_R_d_gammap is valid for R(alpha) and R(eps_p_eq), only for saturated alpha I would see an issue
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
			case enums::exponent_exponential_hard_stress:
				AssertThrow(false, ExcMessage("get_d_R_d_gammap<< exponent_exponential_hard_stress not yet with ay tangents."));
				return 0.;
			case enums::K_alpha_exp:
				return /*d_R_d_gammap =*/ - cm[enums::K] * cm[enums::K_exp] * std::pow(alpha_k + 1e-20, cm[enums::K_exp]-1.) * std::sqrt(2./3.);
//	 		case enums::your_hard_law:
//				return /*d_R_d_gammap =*/ ...;
		}
		return false;
	 }
	 //#######################################################################################################################################################
	 // @todo not compatible to damage, update + Phi_k for all terms
	 /**
	  * @note Be aware of the usage of Phi_k, because the stress norm in the equations can only be
	  * replaced by the current yield stress and yield function, where we here assume the yield function
	  * to be not satisfied (because we iterate)
	  * @param gamma
	  * @param hardStress_R
	  * @param alpha_k
	  * @param Phi_k
	  * @param n_k
	  * @param stress_T_t
	  * @param alpha_n
	  * @param cm
	  * @param HillT_H
	  * @param dmg_mu
	  * @param dmg_p
	  * @return
	  */
	 template<typename Number>
	 Number get_dPhi_dgamma( const Number &gamma, const Number &hardStress_R, const Number &alpha_k, const Number &Phi_k, const SymmetricTensor<2,3,Number> &n_k,
			 	 	 	     const SymmetricTensor<2,3,Number> &stress_T_t, const double &alpha_n,
							 const std::vector<double> &cm, const SymmetricTensor<4,3> &HillT_H, const Number &dmg_mu=1., const Number &dmg_p=1. )
	 {
		 if ( cm[enums::P_hard_iso_kin] < 1 )
			 AssertThrow( false, ExcMessage("elpl_equation - get_dPhi_dgamma<< Ay tangent not yet for kinematic hardening."));

		SymmetricTensor<4,3,Number> A_inv = get_Ainv( gamma, hardStress_R, Number(0), cm, HillT_H, dmg_mu, dmg_p );
		// The use of the newest yield function seems to be quite quite advantages (reduces nbr of qp iterations by one,
		// and for linear isotropic hardening instead of five iterations, we get the desired one-step solution)
		 if ( cm[enums::plastic_aniso] > 0.5 ) // true
		 {
			return - n_k  * (
								(A_inv*A_inv)
								* HillT_H
								* 2. * cm[enums::mu] / ( std::sqrt(2./3.) * (cm[enums::yield_stress]-hardStress_R) + Phi_k )
								* ( 1. + gamma / ( std::sqrt(1.5)*Phi_k + (cm[enums::yield_stress]-hardStress_R) ) * get_d_R_d_gammap(gamma, alpha_k, alpha_n, cm) )
							)
						  * stress_T_t
				   + std::sqrt(2./3.) * get_d_R_d_gammap(gamma, alpha_k, alpha_n, cm);
		 }
		 else
			 return - 2. * cm[enums::mu] + std::sqrt(2./3.) * get_d_R_d_gammap(gamma, alpha_k, alpha_n, cm);
	 }
	 //#######################################################################################################################################################
	 template<typename Number>
	 Number get_hardeningStress_R( const Number &alpha_k, const std::vector<double> &cm )
	 {
		switch ( int(cm[enums::plastic_hardening]) )
		{
			case enums::standard_lin_iso:
				return /*hardStress_R =*/ - cm[enums::P_hard_iso_kin] * cm[enums::K] * alpha_k;
			case enums::saturated_alpha:
				return /*hardStress_R =*/ - cm[enums::P_hard_iso_kin] * cm[enums::K] * alpha_k;
			case enums::saturated_Voce_hard_stress:
				return /*hardStress_R =*/ - cm[enums::P_hard_iso_kin] * cm[enums::yield_stress_incr] * ( 1. - std::exp(-cm[enums::K] / cm[enums::yield_stress_incr] * alpha_k) );
			case enums::saturated_Miehe_hard_stress:
				return /*hardStress_R =*/ - cm[enums::P_hard_iso_kin] * ( cm[enums::K] * alpha_k
																		   + cm[enums::yield_stress_incr] * (1. - std::exp(-cm[enums::K_exp] * alpha_k) ) );
			case enums::exponent_exponential_hard_stress:
				return /*hardStress_R =*/ - cm[enums::P_hard_iso_kin] * ( cm[enums::K] * std::pow( alpha_k, 1.5)
																		   + cm[enums::yield_stress_incr] * (1. - std::exp(-cm[enums::K_exp] * alpha_k) ) );
			case enums::K_alpha_exp:
				return /*hardStress_R =*/ - cm[enums::K] * std::pow(alpha_k, cm[enums::K_exp]);
//	 		case enums::your_hard_law:
//				return /*hardStress_R =*/ ...;
		}
		return false;
	 }
	 //#######################################################################################################################################################
	 template<typename Number>
	 SymmetricTensor<2,3,Number> get_backStress_B( const SymmetricTensor<2,3,Number> &beta_k, const std::vector<double> &cm )
	 {
		//switch ( int(cm[enums::plastic_kin_hardening]) )
		switch ( enums::P_kinHard_standard_lin ) // HARDCODED
		{
			case enums::P_kinHard_standard_lin:
				return /*backStress_B =*/ - 2./3. * cm[enums::K] * ( 1.-cm[enums::P_hard_iso_kin] ) * beta_k;
			case enums::P_kinHard_saturated:
				return /*backStress_B =*/ - 2./3. * cm[enums::K] * ( 1.-cm[enums::P_hard_iso_kin] ) * beta_k / ( beta_k.norm() + 1e-20 ); // WAG
//	 		case enums::your_hard_law:
//				return /*backStress_B =*/ ...;
		}
		return SymmetricTensor<2,3,Number>();
	 }
	 //#######################################################################################################################################################
	 template<typename Number>
	 Number get_yielding_norm( const SymmetricTensor<2,3,Number> &tensor, const SymmetricTensor<4,3> &HillT_H, const bool &GG_mode_active=false, bool &GG_mode_requested=false )
	 {
		// @todo-ensure: How can the Hill-norm get negative?
		Number tmp = tensor * HillT_H * tensor;
		if ( SacadoQP::get_value(tmp) < 0 )
		{
			// If the GG-mode is active, then we request a restart (=true) and overwrite the negative value by just something.
			if ( GG_mode_active )
			{
				GG_mode_requested=true;
				std::cout << "get_yielding_norm<< The Hill norm of the stress tensor got negative as " << tmp << "." << std::endl;
				tmp = 9e9;
			}
			else
				AssertThrow( false, ExcMessage("elpl_equation_list<< The Hill norm of the stress tensor got negative as "+std::to_string(SacadoQP::get_value(tmp))+"."));
		}
		// Overwrite derivatives with truly zero
		else if ( SacadoQP::get_value(tmp) == 0 )
			tmp = 0.;
		return std::sqrt(tmp);
	 }
	 //#######################################################################################################################################################
	 template<typename Number>
	 SymmetricTensor<2,3,Number> get_n_n1( SymmetricTensor<2,3,Number> &stress_k, const SymmetricTensor<4,3> &HillT_H, const bool &GG_mode_active=false, bool &GG_mode_requested=false )
	 {
		// For zero strain (initial) we get zero stress and 0/0 doesn't work
		 Number denominator = (SacadoQP::get_value( stress_k.norm() )==0.) ? 1e-100 : get_yielding_norm( stress_k, HillT_H, GG_mode_active, GG_mode_requested );
		return (HillT_H * stress_k) / denominator;
	 }
	 //#######################################################################################################################################################
	 // @todo We run into trouble here with our optional arguments, I would like to be able to use GG-mode options also without the damage option.
	 // As a cheap workaround we used another function here without the damage
	 template<typename Number>
	 Number get_plastic_yield_fnc ( const SymmetricTensor<2,3,Number> &stress_k, const Number &hardStress_R,
									const std::vector<double> &cm, const SymmetricTensor<4,3> &HillT_H,
									const Number &dmg_p=1., const bool &GG_mode_active=false, bool &GG_mode_requested=false )
	 {
		// Note that for the following to work, namely that we don't use alpha as an input argument, we need to compute the newest alpha
		// via the get_alpha_n_k_ep function (which writes it into the member variable \a alpha_k)
		 return 1./dmg_p * get_yielding_norm( stress_k, HillT_H, GG_mode_active, GG_mode_requested ) - std::sqrt(2./3.) * ( cm[enums::yield_stress] - hardStress_R );
	 }
	 //#######################################################################################################################################################
	 template<typename Number>
	 Number get_plastic_yield_fnc ( const SymmetricTensor<2,3,Number> &stress_k, const Number &hardStress_R,
									const std::vector<double> &cm, const SymmetricTensor<4,3> &HillT_H,
									const bool &GG_mode_active=false, bool &GG_mode_requested=false )
	 {
		// Note that for the following to work, namely that we don't use alpha as an input argument, we need to compute the newest alpha
		// via the get_alpha_n_k_ep function (which writes it into the member variable \a alpha_k)
		 return get_yielding_norm( stress_k, HillT_H, GG_mode_active, GG_mode_requested ) - std::sqrt(2./3.) * ( cm[enums::yield_stress] - hardStress_R );
	 }
	 //#######################################################################################################################################################
	 // The following functions are only used for analytical tangents (because it's the analytical tangent), hence we need to \a Number template
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
	  * @note Only valid for pure elasticity
	  */
	 SymmetricTensor<4,3> get_tangent_elastic( const std::vector<double> &cm )
	 {
		return cm[enums::kappa] * outer_product(unit_symmetric_tensor<3>(), unit_symmetric_tensor<3>())
			   + 2. * cm[enums::mu] * deviator_tensor<3>();
	 }
	 //#######################################################################################################################################################
	 /**
	  * @note Only valid for pure elasto-plasticity (not for damage)
	  */
	 SymmetricTensor<4,3> get_tangent_plastic( const SymmetricTensor<2,3> &stress_n1, const SymmetricTensor<2,3> &stress_T_t, const double &gamma, const SymmetricTensor<2,3> &n_n1,
											   const double &alpha_k, const double &alpha_n, const double &hardStress_R, const double &Phi_k,
											   const std::vector<double> &cm, const SymmetricTensor<4,3> &HillT_H )
	 {
		 if ( cm[enums::P_hard_iso_kin] < 1 )
			 AssertThrow( false, ExcMessage("elpl_equation - get_dPhi_dgamma<< Ay tangent not yet for kinematic hardening."));

		 if ( cm[enums::plastic_aniso] > 0.5 ) // true
		 {
			SymmetricTensor<4,3> N_four = get_N_four( stress_n1, HillT_H );
			SymmetricTensor<4,3> d_Tt_d_eps = cm[enums::kappa] * outer_product( unit_symmetric_tensor<3>(), unit_symmetric_tensor<3>() )
											  + 2. * cm[enums::mu] * deviator_tensor<3>();
			SymmetricTensor<4,3> E_e = invert( invert(d_Tt_d_eps) + gamma * N_four );

			//std::cout << "check 0 " << invert(invert(E_e))-E_e << std::endl;
			//std::cout << "check 1 " << E_e * (invert(d_Tt_d_eps) + gamma * N_four) << std::endl;

			return E_e - 1. / ( n_n1 * E_e * n_n1 - std::sqrt(2./3.) * get_d_R_d_gammap(gamma, alpha_k, alpha_n, cm) ) * ( outer_product(E_e*n_n1, n_n1*E_e) );
		 }
		 else
		 {
			return cm[enums::kappa] * outer_product(unit_symmetric_tensor<3>(), unit_symmetric_tensor<3>())
				   + 2. * cm[enums::mu] * deviator_tensor<3>()
				   - 4.*cm[enums::mu]*cm[enums::mu] / -(get_dPhi_dgamma( gamma, hardStress_R, alpha_k, Phi_k, n_n1, stress_T_t, alpha_n, cm, HillT_H )) * outer_product(n_n1,n_n1)
				   - 4.*cm[enums::mu]*cm[enums::mu] * gamma / deviator(stress_T_t).norm() * ( deviator_tensor<3>() - outer_product(n_n1,n_n1) );
		 }
	 }
 }


