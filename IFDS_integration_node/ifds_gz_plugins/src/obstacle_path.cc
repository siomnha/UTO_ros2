#include <chrono>
#include <cmath>
#include <memory>

#include <gz/math/Pose3.hh>
#include <gz/math/Quaternion.hh>
#include <gz/math/Vector3.hh>
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
/// Drive a kinematic model between two points at constant linear speed.
class ObstaclePath final :
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
    const auto initialPose = gz::sim::worldPose(entity, ecm);
    this->start = sdf->Get<gz::math::Vector3d>("start", initialPose.Pos()).first;
    this->end = sdf->Get<gz::math::Vector3d>("end", initialPose.Pos()).first;
    this->velocity = std::abs(sdf->Get<double>("velocity", 0.0).first);
    this->direction = this->end - this->start;
    this->length = this->direction.Length();
    if (this->length > 0.0) {
      this->direction /= this->length;
    }
    this->orientation = initialPose.Rot();
  }

  void PreUpdate(
    const gz::sim::UpdateInfo & info,
    gz::sim::EntityComponentManager & ecm) override
  {
    if (info.paused || !this->model.Valid(ecm) || this->length <= 0.0 ||
      this->velocity <= 0.0)
    {
      return;
    }

    const double time = std::chrono::duration<double>(info.simTime).count();
    const double periodDistance = 2.0 * this->length;
    const double wrappedDistance = std::fmod(this->velocity * time, periodDistance);
    const double distance = wrappedDistance <= this->length ?
      wrappedDistance : periodDistance - wrappedDistance;
    this->model.SetWorldPoseCmd(
      ecm, gz::math::Pose3d(this->start + distance * this->direction, this->orientation));
  }

private:
  gz::sim::Model model{gz::sim::kNullEntity};
  gz::math::Vector3d start;
  gz::math::Vector3d end;
  gz::math::Vector3d direction;
  gz::math::Quaterniond orientation;
  double velocity{0.0};
  double length{0.0};
};
}  // namespace ifds::sim

GZ_ADD_PLUGIN(
  ifds::sim::ObstaclePath,
  gz::sim::System,
  gz::sim::ISystemConfigure,
  gz::sim::ISystemPreUpdate)
