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
#include "nlobe.h"

namespace mujoco::plugin::sdf {
namespace {

static mjtNum sector(const mjtNum p[2], const mjtNum radius, const mjtNum sector_angle) {
  // a circle centered at (0, 0), with a sector cut out
  mjtNum sdf_circle = mju_norm(p, 2) - radius;
  mjtNum b = mjPI - sector_angle;
  // don't use mju_sign here to avoid issues when p.y == 0
  mjtNum nearest_arc[2] = {-mju_cos(b), p[1] > 0.0 ? mju_sin(b) : - mju_sin(b)};

  mju_scl(nearest_arc, nearest_arc, radius, 2);
  mju_sub(nearest_arc, nearest_arc, p, 2);
  
  mjtNum sdf_segment = - mju_norm(nearest_arc, 2);
  mjtNum cos_p = p[0] / mjMAX(mju_norm(p, 2), mjMINVAL);
  return cos_p > mju_cos(sector_angle) ? sdf_circle : sdf_segment;
}

static mjtNum gradSector(mjtNum grad[3], const mjtNum p[2], const mjtNum radius, const mjtNum sector_angle) {
  mjtNum sdf_circle = mju_norm(p, 2) - radius;
  mjtNum b = mjPI - sector_angle;
  mjtNum nearest_arc[2] = {-mju_cos(b), p[1] > 0.0 ? mju_sin(b) : - mju_sin(b)};

  mju_sub(nearest_arc, nearest_arc, p, 2);
  mjtNum len_xy = mjMAX(mju_norm(p, 2), mjMINVAL);
  mjtNum sdf_segment = - mjMAX(mju_norm(nearest_arc, 2), mjMINVAL);
  mjtNum cos_p = p[0] / len_xy;

  bool in_circle = cos_p > mju_cos(sector_angle);
  grad[0] = in_circle ? p[0] / len_xy : nearest_arc[0] / abs(sdf_segment);
  grad[1] = in_circle ? p[1] / len_xy : nearest_arc[1] / abs(sdf_segment);

  return in_circle ? sdf_circle : sdf_segment;
}

static mjtNum distance(const mjtNum p[3], const mjtNum attributes[3]) {
  // lobe is offset by the lobe radius

  // We can use a naive rotation repetition, since the SDFs are just circles.
  // https://iquilezles.org/articles/sdfrepetition/
  mjtNum sector_angle = 2 * mjPI / attributes[0];
  mjtNum an = mju_atan2(p[1], p[0]);
  mjtNum i = round(an / sector_angle);

  mjtNum c = sector_angle * i;
  // rot is a clockwise rotation
  mjtNum rot[4] = {mju_cos(c), mju_sin(c), -mju_sin(c), mju_cos(c)};
  mjtNum pxy[2] = {p[0], p[1]};

  // Rotate p into local coordinates
  mjtNum pxy_rot[2];
  mju_mulMatVec(pxy_rot, rot, pxy, 2, 2);
  mjtNum offset[2] = {attributes[1], 0.0};
  mju_sub(pxy_rot, pxy_rot, offset, 2);
  mjtNum sdf_nlobe_2d = sector(pxy_rot, attributes[1], sector_angle);
  return Extrude(p, sdf_nlobe_2d, attributes[2]);
}

static void gradient(mjtNum grad[3], const mjtNum p[3], const mjtNum attributes[3]) {
  mjtNum sector_angle = 2 * mjPI / attributes[0];
  mjtNum an = mju_atan2(p[1], p[0]);
  mjtNum i = round(an / sector_angle);

  mjtNum c = sector_angle * i;
  mjtNum rot[4] = {mju_cos(c), mju_sin(c), -mju_sin(c), mju_cos(c)};
  mjtNum pxy[2] = {p[0], p[1]};

  // Rotate p into local coordinates
  mjtNum pxy_rot[2];
  mju_mulMatVec(pxy_rot, rot, pxy, 2, 2);
  mjtNum offset[2] = {attributes[1], 0.0};
  mju_sub(pxy_rot, pxy_rot, offset, 2);
  mjtNum grad_rot[2];
  mjtNum sdf_nlobe_2d = gradSector(grad_rot, pxy_rot, attributes[1], sector_angle);
  // We now need to rotate the gradient back to global coordinates
  // Invert the rotation matrix by taking the transpose
  mju_mulMatTVec(grad, rot, grad_rot, 2, 2);
  gradExtrude(grad, p, sdf_nlobe_2d, attributes[2]);
}


}  // namespace

// factory function
std::optional<NLobe> NLobe::Create(
    const mjModel* m, mjData* d, int instance) {
  if (CheckAttr("nlobes", m, instance) && CheckAttr("loberadius", m, instance) && CheckAttr("height", m, instance)) {
    return NLobe(m, d, instance);
  } else {
    mju_warning("Invalid parameters in NLobe plugin");
    return std::nullopt;
  }
}

// plugin constructor
NLobe::NLobe(const mjModel* m, mjData* d, int instance) {
  SdfDefault<NLobeAttribute> defattribute;

  for (int i=0; i < NLobeAttribute::nattribute; i++) {
    attribute[i] = defattribute.GetDefault(
        NLobeAttribute::names[i],
        mj_getPluginConfig(m, instance, NLobeAttribute::names[i]));
  }
}

// sdf
mjtNum NLobe::Distance(const mjtNum point[3]) const {
  return distance(point, attribute);
}

// gradient of sdf
void NLobe::Gradient(mjtNum grad[3], const mjtNum p[3]) const {
  gradient(grad, p, attribute);
}

// plugin registration
void NLobe::RegisterPlugin() {
  mjpPlugin plugin;
  mjp_defaultPlugin(&plugin);

  plugin.name = "mujoco.sdf.nlobe";
  plugin.capabilityflags |= mjPLUGIN_SDF;

  plugin.nattribute = NLobeAttribute::nattribute;
  plugin.attributes = NLobeAttribute::names;
  plugin.nstate = +[](const mjModel* m, int instance) { return 0; };

  plugin.init = +[](const mjModel* m, mjData* d, int instance) {
    auto sdf_or_null = NLobe::Create(m, d, instance);
    if (!sdf_or_null.has_value()) {
      return -1;
    }
    d->plugin_data[instance] = reinterpret_cast<uintptr_t>(
        new NLobe(std::move(*sdf_or_null)));
    return 0;
  };
  plugin.destroy = +[](mjData* d, int instance) {
    delete reinterpret_cast<NLobe*>(d->plugin_data[instance]);
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
        auto* sdf = reinterpret_cast<NLobe*>(d->plugin_data[instance]);
        return sdf->Distance(point);
      };
  plugin.sdf_gradient = +[](mjtNum gradient[3], const mjtNum point[3],
                        const mjData* d, int instance) {
    auto* sdf = reinterpret_cast<NLobe*>(d->plugin_data[instance]);
    sdf->Gradient(gradient, point);
  };
  plugin.sdf_staticdistance =
      +[](const mjtNum point[3], const mjtNum* attributes) {
        return distance(point, attributes);
      };
  plugin.sdf_aabb =
      +[](mjtNum aabb[6], const mjtNum* attributes) {
        aabb[0] = aabb[1] = aabb[2] = 0;
        aabb[3] = aabb[4] = 2 * attributes[1];
        aabb[5] = attributes[2] / 2.0;
      };
  plugin.sdf_attribute =
      +[](mjtNum attribute[], const char* name[], const char* value[]) {
        SdfDefault<NLobeAttribute> defattribute;
        defattribute.GetDefaults(attribute, name, value);
      };

  mjp_registerPlugin(&plugin);
}

}  // namespace mujoco::plugin::sdf
