#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_MATERIALS_SHEENLTC_H
#define PBRT_MATERIALS_SHEENLTC_H

// materials/sheenltc.h*
#include "pbrt.h"
#include "material.h"

namespace pbrt {

// SheenLTCMaterial Declarations
class SheenLTCMaterial : public Material {
  public:
    // SheenLTCMaterial Public Methods
    SheenLTCMaterial(const std::string &type,
                     const std::shared_ptr<Texture<Spectrum>> &Csheen,
                     const std::shared_ptr<Texture<Float>> &alpha,
                     const std::shared_ptr<Texture<Float>> &bumpMap)
        : type(type), Csheen(Csheen), alpha(alpha), bumpMap(bumpMap) {}
    void ComputeScatteringFunctions(SurfaceInteraction *si, MemoryArena &arena,
                                    TransportMode mode,
                                    bool allowMultipleLobes) const;

  private:
    // SheenLTCMaterial Private Data
    std::string type;
    std::shared_ptr<Texture<Spectrum>> Csheen;
    std::shared_ptr<Texture<Float>> alpha, bumpMap;
};

SheenLTCMaterial *CreateSheenLTCMaterial(const TextureParams &mp);

}  // namespace pbrt

#endif  // PBRT_MATERIALS_SHEENLTC_H
