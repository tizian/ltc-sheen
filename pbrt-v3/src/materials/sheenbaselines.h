#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_MATERIALS_SHEENBASELINES_H
#define PBRT_MATERIALS_SHEENBASELINES_H

// materials/sheenbaselines.h*
#include "pbrt.h"
#include "material.h"

namespace pbrt {

// SheenBaselinesMaterial Declarations
class SheenBaselinesMaterial : public Material {
  public:
    // SheenBaselinesMaterial Public Methods
    SheenBaselinesMaterial(const std::string &type,
                           const std::shared_ptr<Texture<Spectrum>> &Csheen,
                           const std::shared_ptr<Texture<Float>> &alpha,
                           const std::shared_ptr<Texture<Float>> &bumpMap)
        : type(type), Csheen(Csheen), alpha(alpha), bumpMap(bumpMap) {}
    void ComputeScatteringFunctions(SurfaceInteraction *si, MemoryArena &arena,
                                    TransportMode mode,
                                    bool allowMultipleLobes) const;

  private:
    // SheenBaselinesMaterial Private Data
    std::string type;
    std::shared_ptr<Texture<Spectrum>> Csheen;
    std::shared_ptr<Texture<Float>> alpha, bumpMap;
};

SheenBaselinesMaterial *CreateSheenBaselinesMaterial(const TextureParams &mp);

}  // namespace pbrt

#endif  // PBRT_MATERIALS_SHEENBASELINES_H
