#include <iostream>
#include <cmath>

#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
 
#include "pinocchio/autodiff/casadi.hpp"
#include <casadi/casadi.hpp>

int main()
{
  // Short name for convenience
  namespace pin = pinocchio;

  // Define types to be used later
  typedef double Scalar;
  typedef casadi::SX ADScalar;

  typedef pin::ModelTpl<Scalar> Model;
  typedef Model::Data Data;

  typedef pin::ModelTpl<ADScalar> ADModel;
  typedef ADModel::Data ADData;

  typedef Model::ConfigVectorType ConfigVector;
  typedef Model::TangentVectorType TangentVector;

  typedef ADModel::ConfigVectorType ConfigVectorAD;
  typedef ADModel::TangentVectorType TangentVectorAD;

  // Path to URDF file
  const std::string urdf_filename = "/home/shamil/Desktop/phd/code/beam_insertion/"
                                    "beam_ws/src/setup_description/urdf/setup_descr_pend_fixed.urdf";

  // Load the model from urdf
  Model model;
  pin::urdf::buildModel(urdf_filename, model);
  std::cout << "model name: " << model.name << std::endl;
  // Create data required by the algorithms
  pin::Data data(model);
  const std::string aprx = "pend";

  // Get EE frame ID for forward kinematics
  const std::string ee_link_name = "panda_link8";
  const std::string beam_link_name = "pendulum_rod";
  pin::Model::Index ee_frame_id = model.getFrameId(ee_link_name);
  pin::Model::Index beam_frame_id = model.getFrameId(beam_link_name);
  std::cout << "EE frame ID: " << ee_frame_id << std::endl;
  std::cout << "Beam frame ID: " << beam_frame_id << std::endl;

  // Set a configuration, joint velocities and accelerations
  ConfigVector q(model.nq);
  q << -M_PI_2, -M_PI/6, 0.0, -2*M_PI/3, 0.0, M_PI_2, M_PI_4;
  TangentVector v(TangentVector::Random(model.nv));
  TangentVector a(TangentVector::Random(model.nv));
  std::cout << "q: " << q.transpose() << std::endl;
  std::cout << "v: " << v.transpose() << std::endl;
  std::cout << "a: " << a.transpose() << std::endl;

  // Perform the forward kinematics over the kinematic tree
  pin::forwardKinematics(model, data, q, v, a);
  pin::updateFramePlacements(model, data);
 
  Eigen::VectorXd p_ee = data.oMf[ee_frame_id].translation();
  auto R_ee = data.oMf[ee_frame_id].rotation();
  Eigen::VectorXd p_beam = data.oMf[beam_frame_id].translation();
  auto R_beam = data.oMf[beam_frame_id].rotation();
  auto v_ee =  pin::getFrameVelocity(model, data, 
                          ee_frame_id, pin::LOCAL_WORLD_ALIGNED);
  auto v_beam =  pin::getFrameVelocity(model, data, 
                          beam_frame_id, pin::LOCAL_WORLD_ALIGNED);
  auto a_ee = pin::getFrameAcceleration(model, data, 
                          ee_frame_id, pin::LOCAL_WORLD_ALIGNED);

  std::cout << "EE position num: " << p_ee.transpose() << std::endl;
  std::cout << "EE orientation num: " << R_ee << std::endl;
  std::cout << "Beam position num: " << p_beam.transpose() << std::endl;
  std::cout << "Beam orientation num: " << R_beam << std::endl;
  std::cout << "EE velocity num: " << v_ee.linear().transpose() << " "
                                  << v_ee.angular().transpose() << std::endl;
  std::cout << "Beam velocity num: " << v_beam.linear().transpose() << " "
                                  << v_beam.angular().transpose() << std::endl;
  std::cout << "EE accel num: " << a_ee.linear().transpose() << " "
                                  << a_ee.angular().transpose() << std::endl;

  // Initialize symbolic model
  ADModel ad_model = model.cast<ADScalar>();
  ADData ad_data(ad_model);

  // Define symbolic variables for q, dq, ddq
  casadi::SX cs_q = casadi::SX::sym("q", model.nq);
  ConfigVectorAD q_ad(model.nq);
  pin::casadi::copy(cs_q,q_ad);
  
  casadi::SX cs_v = casadi::SX::sym("v", model.nv);
  TangentVectorAD v_ad(model.nv);
  pin::casadi::copy(cs_v,v_ad);
  
  casadi::SX cs_a = casadi::SX::sym("a", model.nv);
  TangentVectorAD a_ad(model.nv);
  pin::casadi::copy(cs_a,a_ad);

  // Evaluate forward kinematics on symbolic model
  pin::forwardKinematics(ad_model, ad_data, q_ad, v_ad, a_ad);
  pin::updateGlobalPlacements(ad_model, ad_data);
  pin::updateFramePlacements(ad_model, ad_data);

  auto ad_v_ee =  pin::getFrameVelocity(ad_model, ad_data, 
                          ee_frame_id, pin::LOCAL_WORLD_ALIGNED);
  auto ad_a_ee =  pin::getFrameAcceleration(ad_model, ad_data, 
                          ee_frame_id, pin::LOCAL_WORLD_ALIGNED);

  auto ad_v_beam =  pin::getFrameVelocity(ad_model, ad_data, 
                          beam_frame_id, pin::LOCAL_WORLD_ALIGNED);
  auto ad_a_beam =  pin::getFrameAcceleration(ad_model, ad_data, 
                          beam_frame_id, pin::LOCAL_WORLD_ALIGNED);

  pin::computeJointJacobians(ad_model, ad_data, q_ad);
  pin::computeJointJacobiansTimeVariation(ad_model, ad_data, q_ad, v_ad);
  pin::updateGlobalPlacements(ad_model, ad_data);
  pin::updateFramePlacements(ad_model, ad_data);

  auto ad_jac_beam = pin::getFrameJacobian(ad_model, ad_data, 
                          beam_frame_id, pin::LOCAL_WORLD_ALIGNED);

  ADData::Matrix6x ad_djac_beam(6, ad_model.nv); ad_djac_beam.fill(0.);
  pin::getFrameJacobianTimeVariation(ad_model, ad_data, 
                          beam_frame_id, pin::LOCAL_WORLD_ALIGNED, ad_djac_beam);

  // Get a symbolic expression for ee position
  casadi::SX cs_p_ee(3,1);
  casadi::SX cs_R_ee(3,3);
  casadi::SX cs_v_ee(6,1);
  casadi::SX cs_a_ee(6,1);

  casadi::SX cs_p_beam(3,1);
  casadi::SX cs_R_beam(3,3);
  casadi::SX cs_v_beam(6,1);
  casadi::SX cs_a_beam(6,1);
  for (Eigen::DenseIndex k = 0; k < 3; ++k){
    cs_p_ee(k) = ad_data.oMf[ee_frame_id].translation()[k];
    cs_p_beam(k) = ad_data.oMf[beam_frame_id].translation()[k];
    cs_v_ee(k) = ad_v_ee.linear()[k];
    cs_v_ee(3+k) = ad_v_ee.angular()[k];
    cs_v_beam(k) = ad_v_beam.linear()[k];
    cs_v_beam(3+k) = ad_v_beam.angular()[k];
    cs_a_ee(k) = ad_a_ee.linear()[k];
    cs_a_ee(3+k) = ad_a_ee.angular()[k];
    cs_a_beam(k) = ad_a_beam.linear()[k];
    cs_a_beam(3+k) = ad_a_beam.angular()[k];
    for (Eigen::DenseIndex j = 0; j < 3; ++j){
      cs_R_ee(k,j) = ad_data.oMf[ee_frame_id].rotation()(k,j);
      cs_R_beam(k,j) = ad_data.oMf[beam_frame_id].rotation()(k,j);
    }
  }

  casadi::SX cs_jac_beam(6,7);
  casadi::SX cs_djac_beam(6,7);
  for (Eigen::DenseIndex k = 0; k < 6; ++k){
    for (Eigen::DenseIndex j = 0; j < 7; ++j){
      cs_jac_beam(k,j) = ad_jac_beam(k,j);
      cs_djac_beam(k,j) = ad_djac_beam(k,j);
    }
  }

  // A casadi function for evaluating kinematic. Define a variable that specifies
  // which beam model approximation is used
  casadi::Function eval_pee("eval_pee",
                            casadi::SXVector {cs_q},
                            casadi::SXVector {cs_p_ee});
  eval_pee.save("../../beam_insertion_py/casadi_fcns/eval_pee.casadi");

  casadi::Function eval_Ree("eval_Ree",
                            casadi::SXVector {cs_q},
                            casadi::SXVector {cs_R_ee});
  eval_Ree.save("../../beam_insertion_py/casadi_fcns/eval_Ree.casadi");

  casadi::Function eval_vee("eval_vee",
                            casadi::SXVector {cs_q, cs_v},
                            casadi::SXVector {cs_v_ee});
  eval_vee.save("../../beam_insertion_py/casadi_fcns/eval_vee.casadi");

  casadi::Function eval_aee("eval_aee",
                            casadi::SXVector {cs_q, cs_v, cs_a},
                            casadi::SXVector {cs_a_ee});
  eval_aee.save("../../beam_insertion_py/casadi_fcns/eval_aee.casadi");


  casadi::Function eval_pbeam("eval_p" + aprx,
                            casadi::SXVector {cs_q},
                            casadi::SXVector {cs_p_beam});
  eval_pbeam.save("../../beam_insertion_py/casadi_fcns/eval_p" + aprx + ".casadi");

  casadi::Function eval_Rbeam("eval_R" + aprx,
                            casadi::SXVector {cs_q},
                            casadi::SXVector {cs_R_beam});
  eval_Rbeam.save("../../beam_insertion_py/casadi_fcns/eval_R" + aprx + ".casadi");

  casadi::Function eval_vbeam("eval_v" + aprx,
                            casadi::SXVector {cs_q, cs_v},
                            casadi::SXVector {cs_v_beam});
  // eval_vbeam.save("../../beam_insertion_py/casadi_fcns/eval_vbeam.casadi");

  casadi::Function eval_abeam("eval_a" + aprx,
                            casadi::SXVector {cs_q, cs_v, cs_a},
                            casadi::SXVector {cs_a_beam});
  // eval_abeam.save("../../beam_insertion_py/casadi_fcns/eval_abeam.casadi");

  casadi::Function eval_Jbeam("eval_J" + aprx,
                            casadi::SXVector {cs_q},
                            casadi::SXVector {cs_jac_beam});
  eval_Jbeam.save("../../beam_insertion_py/casadi_fcns/eval_J" + aprx + ".casadi");

  casadi::Function eval_dJbeam("eval_dJ" + aprx,
                            casadi::SXVector {cs_q, cs_v},
                            casadi::SXVector {cs_djac_beam});
  eval_dJbeam.save("../../beam_insertion_py/casadi_fcns/eval_dJ" + aprx + ".casadi");
  
  std::vector<double> q_vec((size_t)model.nq);
  Eigen::Map<ConfigVector>(q_vec.data(),model.nq,1) = q;

  std::vector<double> v_vec((size_t)model.nv);
  Eigen::Map<ConfigVector>(v_vec.data(),model.nv,1) = v;

  std::vector<double> a_vec((size_t)model.nv);
  Eigen::Map<ConfigVector>(a_vec.data(),model.nv,1) = a;

  std::cout << "EE position sym: " << eval_pee(casadi::DMVector {q_vec}) << std::endl;
  std::cout << "EE orientation sym: " << eval_Ree(casadi::DMVector {q_vec}) << std::endl;
  std::cout << "Beam position sym: " << eval_pbeam(casadi::DMVector {q_vec}) << std::endl;
  std::cout << "Beam orientation sym: " << eval_Rbeam(casadi::DMVector {q_vec}) << std::endl;
  std::cout << "EE velocity sym: " << eval_vee(casadi::DMVector {q_vec, v_vec}) << std::endl;
  std::cout << "Beam velocity sym: " << eval_vbeam(casadi::DMVector {q_vec, v_vec}) << std::endl;
  std::cout << "EE accel sym: " << eval_aee(casadi::DMVector {q_vec, v_vec, a_vec}) << std::endl;

  return 0;
}
