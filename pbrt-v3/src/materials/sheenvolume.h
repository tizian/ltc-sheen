#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_MATERIALS_SHEEN_VOLUME_H
#define PBRT_MATERIALS_SHEEN_VOLUME_H

// materials/sheenvolume.h*
#include "pbrt.h"
#include "material.h"

namespace pbrt {

// SheenVolumeMaterial Declarations
class SheenVolumeMaterial : public Material {
  public:
    // SheenVolumeMaterial Public Methods
    SheenVolumeMaterial(int maxBounces,
                        const std::shared_ptr<Texture<Spectrum>> &albedo,
                        const std::shared_ptr<Texture<Float>> &density,
                        const std::shared_ptr<Texture<Float>> &sigma,
                        const std::shared_ptr<Texture<Float>> &bumpMap)
        : maxBounces(maxBounces), albedo(albedo), density(density), sigma(sigma), bumpMap(bumpMap) {}
    void ComputeScatteringFunctions(SurfaceInteraction *si, MemoryArena &arena,
                                    TransportMode mode,
                                    bool allowMultipleLobes) const;

  private:
    // SheenVolumeMaterial Private Data
    int maxBounces;
    std::shared_ptr<Texture<Spectrum>> albedo;
    std::shared_ptr<Texture<Float>> density, sigma, bumpMap;

    /* Hacky solution to allow an unbounded number of random variates in
       `SheenSGGXVolume::f`.
       Every time the BSDF is instantiated (via `ComputeScatteringFunctions`),
       this seed is incremented and passed to the BSDF. This further needs to be
       an atomic as multiple threads will simultanously access this function.

       A cleaner solution would be to pass a pointer to a `Sampler` from the
       integrator to the BSDF. This would however require more substantial
       changes throughout the code base that we want like to avoid here.*/
    static std::atomic<uint64_t> seed;
};

SheenVolumeMaterial *CreateSheenVolumeMaterial(const TextureParams &mp);

}  // namespace pbrt

#endif  // PBRT_MATERIALS_SHEEN_VOLUME_H
