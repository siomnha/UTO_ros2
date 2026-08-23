#include <chrono>
#include <cmath>
#include <memory>

#include <gz/math/Pose3.hh>
#include <gz/plugin/Register.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/EventManager.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Types.hh>
#include <gz/sim/UpdateInfo.hh>
#include <gz/sim/Util.hh>
#include <sdf/Element.hh>

namespace ifds::sim
{
class ObstacleOscillator final :
  public gz::sim::System,
  public gz::sim::ISystemConfigure,
  public gz::sim::ISystemPreUpdate
{
public:
  void Configure(
    const gz::sim::Entity & entity,
    const std::shared_ptr<const sdf::Element> & sdf,
    gz::sim::EntityComponentManager & ecm,
    gz::sim::EventManager &) override
  {
    this->model = gz::sim::Model(entity);
    this->initialPose = gz::sim::worldPose(entity, ecm);
    this->amplitudeY = sdf->Get<double>("amplitude_y", 0.0).first;
    this->angularSpeed = sdf->Get<double>("angular_speed", 0.0).first;
    this->phase = sdf->Get<double>("phase", 0.0).first;
  }

  void PreUpdate(
    const gz::sim::UpdateInfo & info,
    gz::sim::EntityComponentManager & ecm) override
  {
    if (info.paused || !this->model.Valid(ecm)) {
      return;
    }

    const double time = std::chrono::duration<double>(info.simTime).count();
    auto pose = this->initialPose;
    pose.Pos().Y() += this->amplitudeY * std::sin(this->angularSpeed * time + this->phase);
    this->model.SetWorldPoseCmd(ecm, pose);
  }

private:
  gz::sim::Model model{gz::sim::kNullEntity};
  gz::math::Pose3d initialPose;
  double amplitudeY{0.0};
  double angularSpeed{0.0};
  double phase{0.0};
};
}  // namespace ifds::sim

GZ_ADD_PLUGIN(
  ifds::sim::ObstacleOscillator,
  gz::sim::System,
  gz::sim::ISystemConfigure,
  gz::sim::ISystemPreUpdate)
