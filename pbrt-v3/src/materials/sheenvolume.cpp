// materials/sheenvolume.cpp*
#include "materials/sheenvolume.h"
#include "paramset.h"
#include "reflection.h"
#include "sampling.h"
#include "rng.h"
#include "materials/sggx.h"

namespace pbrt {

/*  Implementation of a fiber-like SGGX volume layer, evaluated stochastically
    via a random walk. Part of the supplementary material for

        "Practical Multiple-Scattering Sheen Using Linearly Transformed Cosines"
        by Tizian Zeltner, Brent Burley, and Matt Jen-Yuan Chiang 2022.

    For more details about the implementation, see

        "Additional Progress Towards the Unification of Microfacet and
        Microflake Theories"
        by Dupuy et al. 2016.

    ----------------------------------------------------------------------------

    Note: the `BxDF` interface in pbrt-v3 is not well suited for implementing
    stochastic BSDFs without closed form solutions. In particular, pbrt-v3
    uses an explicit division by the sampling PDF in `integrator.cpp` which
    is not available in our case.

    One solution would be to also use a stochastic PDF and to return a
    consistent result for both `SheenSGGXVolume::f` and `SheenSGGXVolume::Pdf`
    s.t. the two values cancel out to `1` later. But unfortunately, a stochastic
    PDF is not usable for computing MIS weights as their computation involves a
    division that causes bias.

    As a workaround, we implement `SheenSGGXVolume::Sample_f` with very basic
    uniform hemisphere sampling. Interestingly, this is in practice not much
    worse (and sometimes even an improvement!) compared to a more sophisticated
    random walk sampling as it guarantees that at the very least all generated
    `wi` reflect out of the volume layer. In comparison, sampled random paths
    through the volume layer tend to escape towards the bottom which results in
    many invalid samples.

    No matter the choice of sampling routine, we found the resulting BRDF model
    to have too high variance for our use cases. This motivated us to
    approximate the resulting behavior with a simple LTC model instead, see
    `sheenltc.cpp`.
*/

class SheenSGGXVolume : public BxDF {
  public:
    SheenSGGXVolume(int maxBounces,
                    Spectrum albedo,
                    Float density,
                    Float sigma,
                    uint64_t seed)
        : BxDF(BxDFType(BSDF_REFLECTION | BSDF_GLOSSY)), maxBounces(maxBounces),
          albedo(albedo), density(density), sigma(sigma), seed(seed) {}
    Spectrum f(const Vector3f &wo, const Vector3f &wi) const;
    Spectrum Sample_f(const Vector3f &wo, Vector3f *wi, const Point2f &u,
                      Float *pdf, BxDFType *sampledType) const;
    Float Pdf(const Vector3f &wo, const Vector3f &wi) const;
    std::string ToString() const;

  private:
    int maxBounces;
    Spectrum albedo;
    Float density, sigma;
    uint64_t seed;
};

Spectrum SheenSGGXVolume::f(const Vector3f &wo, const Vector3f &wi) const {
    Float cosThetaO = CosTheta(wo),
          cosThetaI = CosTheta(wi);
    if (maxBounces == 0 || cosThetaO < 0 || cosThetaI < 0) return Spectrum(0.f);

    /* We need access to an unbounded number of random numbers for the
       stochastic evaluation. See more details in `sheenvolume.h` about this
       workaround. */
    RNG rng;
    rng.SetSequence(seed);

    /* Compute the ellipsoid that parameterizes the SGGX microflakes. Uses
       the "fiber-like" version from Heitz et al. 2015. See `sggx.h`. */
    sggx::Ellipsoid S = sggx::Ellipsoid::fromFiber(Vector3f(0, 0, 1), sigma);

    Vector3f dir    = -wo;  // Propagation direction
    Float    depth  = 0.f;  // Depth inside the volume (Assume unit thickness. 1: top, 0: bottom)
    int      bounce = 0;    // Keep track of scattering order

    Spectrum beta(1.f);  // Path throughput
    Spectrum value(0.f); // Accumulator for BRDF * cos

    // Volumetric random walk, with next event estimation at each scattering event.
    while (true) {
        // Extinction coefficient depends on projected microflake area.
        Float sigmaT = density*sggx::sigma(-dir, S);

        // Sample distance, and update depth (along optical axis).
        Float distSampled = (sigmaT == 0.f) ? Infinity
                                            : -std::log(1.f - rng.UniformFloat()) / sigmaT;
        depth = depth + distSampled * CosTheta(-dir);

        // Possibly escape the layer.
        if (depth < 0.f || depth > 1.f) break;

        // Next event estimation, i.e. compute shadowing towards outgoing direction wi.
        if (CosTheta(wi) > 0.0) {
            Float distAlongWi = depth / std::abs(CosTheta(wi));
            Float sigmaTAlongWi = density*sggx::sigma(-wi, S);

            Float phaseFunction = sggx::evalPhaseSpecular(-dir, wi, S);
            value += beta * albedo * phaseFunction * std::exp(-distAlongWi*sigmaTAlongWi);
        }

        // Scattering event.
        beta *= albedo;
        dir = sggx::samplePhaseSpecular(-dir, S, rng.UniformFloat(), rng.UniformFloat());

        bounce++;
        if (bounce >= maxBounces && maxBounces > 0) break;
    }

    /* Compared to other rendering systems (e.g. Mitsuba), pbrt-v3 accounts
       for the cosine foreshortening factor in the integrator instead of the
       BSDF. So we cancel it out here. */
    return cosThetaI > 0.f ? value / cosThetaI : Spectrum(0.f);
}

Spectrum SheenSGGXVolume::Sample_f(const Vector3f &wo, Vector3f *wi, const Point2f &u,
                                   Float *pdf, BxDFType *sampledType) const {
    /* Naive uniform hemisphere sampling. At least this is guaranteed to result
       in a reflection direction. */
    *wi = UniformSampleHemisphere(u);
    if (CosTheta(wo) < 0.f || CosTheta(*wi) < 0.f) {
        *pdf = 0.f;
        return Spectrum(0.f);
    }
    *pdf = Pdf(wo, *wi);

    /* This value is actually never used and is instead recomputed in
       `integrator.cpp`. So let's avoid another costly random walk here. */
    // return f(wo, *wi);
    return 1.f;
}

Float SheenSGGXVolume::Pdf(const Vector3f &wo, const Vector3f &wi) const {
    if (CosTheta(wo) < 0.f || CosTheta(wi) < 0.f) {
        return 0.f;
    }
    return UniformHemispherePdf();
}

std::string SheenSGGXVolume::ToString() const {
    return StringPrintf("[ SheenSGGXVolume maxBounces: %d albedo: %s density: %f sigma %f]",
                        maxBounces, albedo.ToString().c_str(), density, sigma);
}


void SheenVolumeMaterial::ComputeScatteringFunctions(SurfaceInteraction *si,
                                                     MemoryArena &arena,
                                                     TransportMode mode,
                                                     bool allowMultipleLobes) const {
    // Perform bump mapping with _bumpMap_, if present
    if (bumpMap) Bump(bumpMap, si);

    // Evaluate textures for _SheenBaselinesMaterial_ material and allocate BRDF
    si->bsdf = ARENA_ALLOC(arena, BSDF)(*si);
    Spectrum a = albedo->Evaluate(*si).Clamp();
    Float d = density->Evaluate(*si),
          s = Clamp(sigma->Evaluate(*si), 0, 1);
    if (!a.IsBlack()) {
        si->bsdf->Add(ARENA_ALLOC(arena, SheenSGGXVolume)(maxBounces, a, d, s, seed));
    }
    seed++;
}

SheenVolumeMaterial *CreateSheenVolumeMaterial(const TextureParams &mp) {
    int maxBounces = mp.FindInt("maxBounces", 16);
    std::shared_ptr<Texture<Spectrum>> albedo = mp.GetSpectrumTexture("albedo", 1.f);
    std::shared_ptr<Texture<Float>> density   = mp.GetFloatTexture("density", 1.f);
    std::shared_ptr<Texture<Float>> sigma     = mp.GetFloatTexture("sigma", 0.5f);
    std::shared_ptr<Texture<Float>> bumpMap   = mp.GetFloatTextureOrNull("bumpmap");
    return new SheenVolumeMaterial(maxBounces, albedo, density, sigma, bumpMap);
}

std::atomic<uint64_t> SheenVolumeMaterial::seed(0);

}  // namespace pbrt
