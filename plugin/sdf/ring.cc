// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdint>
#include <optional>
#include <utility>

#include <mujoco/mjplugin.h>
#include <mujoco/mjtnum.h>
#include <mujoco/mujoco.h>
#include "sdf.h"
#include "ring.h"

namespace mujoco::plugin::sdf {
namespace {

static mjtNum distance(const mjtNum p[3], const mjtNum attributes[3]) {
  mjtNum midpoint = 0.5 * (attributes[0] + attributes[1]);
  mjtNum thickness = attributes[0] - attributes[1];
  mjtNum pxy[2] = {p[0], p[1]};
  mjtNum sdf_circle = mju_norm(pxy, 2) - midpoint;
  mjtNum sdf_ring_2d = Onion(sdf_circle, thickness);
  return Extrude(p, sdf_ring_2d, attributes[2]);
}

static void gradient(mjtNum grad[3], const mjtNum p[3], const mjtNum attributes[3]) {
  mjtNum midpoint = 0.5 * (attributes[0] + attributes[1]);
  mjtNum thickness = attributes[0] - attributes[1];
  mjtNum pxy[2] = {p[0], p[1]};
  mjtNum len_xy = mjMAX(mju_norm(pxy, 2), mjMINVAL);
  mjtNum sdf_circle = len_xy - midpoint;
  // set the gradient of the 2d sdf
  grad[0] = p[0] / len_xy;
  grad[1] = p[1] / len_xy;
  grad[2] = 0;
  mjtNum sdf_ring_2d = gradOnion(grad, sdf_circle, thickness);
  gradExtrude(grad, p, sdf_ring_2d, attributes[2]);
}
}  // namespace

// factory function
std::optional<Ring> Ring::Create(
    const mjModel* m, mjData* d, int instance) {
  if (CheckAttr("outerradius", m, instance) && CheckAttr("innerradius", m, instance) && CheckAttr("height", m, instance)) {
    return Ring(m, d, instance);
  } else {
    mju_warning("Invalid parameters in Ring plugin");
    return std::nullopt;
  }
}

// plugin constructor
Ring::Ring(const mjModel* m, mjData* d, int instance) {
  SdfDefault<RingAttribute> defattribute;

  for (int i=0; i < RingAttribute::nattribute; i++) {
    attribute[i] = defattribute.GetDefault(
        RingAttribute::names[i],
        mj_getPluginConfig(m, instance, RingAttribute::names[i]));
  }
}

// sdf
mjtNum Ring::Distance(const mjtNum point[3]) const {
  return distance(point, attribute);
}

// gradient of sdf
void Ring::Gradient(mjtNum grad[3], const mjtNum p[3]) const {
  gradient(grad, p, attribute);
}

// plugin registration
void Ring::RegisterPlugin() {
  mjpPlugin plugin;
  mjp_defaultPlugin(&plugin);

  plugin.name = "mujoco.sdf.ring";
  plugin.capabilityflags |= mjPLUGIN_SDF;

  plugin.nattribute = RingAttribute::nattribute;
  plugin.attributes = RingAttribute::names;
  plugin.nstate = +[](const mjModel* m, int instance) { return 0; };

  plugin.init = +[](const mjModel* m, mjData* d, int instance) {
    auto sdf_or_null = Ring::Create(m, d, instance);
    if (!sdf_or_null.has_value()) {
      return -1;
    }
    d->plugin_data[instance] = reinterpret_cast<uintptr_t>(
        new Ring(std::move(*sdf_or_null)));
    return 0;
  };
  plugin.destroy = +[](mjData* d, int instance) {
    delete reinterpret_cast<Ring*>(d->plugin_data[instance]);
    d->plugin_data[instance] = 0;
  };
  plugin.reset = +[](const mjModel* m, mjtNum* plugin_state, void* plugin_data,
                     int instance) {
    // do nothing
  };
  plugin.compute =
      +[](const mjModel* m, mjData* d, int instance, int capability_bit) {
        // do nothing;
      };
  plugin.sdf_distance =
      +[](const mjtNum point[3], const mjData* d, int instance) {
        auto* sdf = reinterpret_cast<Ring*>(d->plugin_data[instance]);
        return sdf->Distance(point);
      };
  plugin.sdf_gradient = +[](mjtNum gradient[3], const mjtNum point[3],
                        const mjData* d, int instance) {
    auto* sdf = reinterpret_cast<Ring*>(d->plugin_data[instance]);
    sdf->Gradient(gradient, point);
  };
  plugin.sdf_staticdistance =
      +[](const mjtNum point[3], const mjtNum* attributes) {
        return distance(point, attributes);
      };
  plugin.sdf_aabb =
      +[](mjtNum aabb[6], const mjtNum* attributes) {
        aabb[0] = aabb[1] = aabb[2] = 0;
        aabb[3] = aabb[4] = attributes[0] + attributes[1];
        aabb[5] = attributes[1];
      };
  plugin.sdf_attribute =
      +[](mjtNum attribute[], const char* name[], const char* value[]) {
        SdfDefault<RingAttribute> defattribute;
        defattribute.GetDefaults(attribute, name, value);
      };

  mjp_registerPlugin(&plugin);
}

}  // namespace mujoco::plugin::sdf
