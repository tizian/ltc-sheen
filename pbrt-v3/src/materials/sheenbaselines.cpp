// materials/sheenbaselines.cpp*
#include "materials/sheenbaselines.h"
#include "paramset.h"
#include "reflection.h"
#include "sampling.h"
#include "materials/sggx.h"

namespace pbrt {

/*  Implementation of previous sheen models. Part of the supplementary material
    for

        "Practical Multiple-Scattering Sheen Using Linearly Transformed Cosines"
        by Tizian Zeltner, Brent Burley, and Matt Jen-Yuan Chiang 2022.

    See the various `BxDF` implementations below for descriptions.
*/


/*
    type == "burley"

    The isolated sheen component of the principled BRDF presented in

        "Physically Based Shading at Disney"
        by Brent Burley

    in the Physically-based Shading course at SIGGRAPH 2012.
 */
class SheenBurley : public BxDF {
  public:
    SheenBurley(const Spectrum &Csheen)
        : BxDF(BxDFType(BSDF_REFLECTION | BSDF_GLOSSY)), Csheen(Csheen) {}
    Spectrum f(const Vector3f &wo, const Vector3f &wi) const;
    std::string ToString() const;

  private:
    Spectrum Csheen;
};

Spectrum SheenBurley::f(const Vector3f &wo, const Vector3f &wi) const {
    Float cosThetaO = CosTheta(wo),
          cosThetaI = CosTheta(wi);
    if (cosThetaO < 0 || cosThetaI < 0) return 0.f;

    Vector3f wh = wi + wo;
    if (wh.x == 0 && wh.y == 0 && wh.z == 0) return Spectrum(0.);
    wh = Normalize(wh);
    Float cosThetaD = Dot(wi, wh);

    auto schlick = [](Float cosTheta) {
        Float m = Clamp(1 - cosTheta, 0, 1);
        return (m * m) * (m * m) * m;
    };

    return Csheen * schlick(cosThetaD);
}

std::string SheenBurley::ToString() const {
    return StringPrintf("[ SheenBurley Csheen: %s]", Csheen.ToString().c_str());
}


/*
    type == "neubelt_pettineo"

    Sheen model presented in

        "Crafting a Next-Gen Material Pipeline for The Order: 1886"
        by Neubelt and Pettineo

    in the Physically-based Shading course at SIGGRAPH 2013.

    This is a modified version of a previous model from

        "A Microfacet-based BRDF Generator"
        by Ashikhmin et al. 2000.
 */
class SheenNeubeltPettineo : public BxDF {
  public:
    SheenNeubeltPettineo(const Spectrum &Csheen, Float alpha)
        : BxDF(BxDFType(BSDF_REFLECTION | BSDF_GLOSSY)), Csheen(Csheen), alpha(alpha) {}
    Spectrum f(const Vector3f &wo, const Vector3f &wi) const;
    std::string ToString() const;

  private:
    Spectrum Csheen;
    Float alpha;
};

Spectrum SheenNeubeltPettineo::f(const Vector3f &wo, const Vector3f &wi) const {
    Vector3f wh = wi + wo;
    if (wh.x == 0 && wh.y == 0 && wh.z == 0) return Spectrum(0.);
    wh = Normalize(wh);

    Float alpha2 = alpha*alpha;

    Float cosThetaO  = CosTheta(wo),
          cosThetaI  = CosTheta(wi),
          cosThetaH  = CosTheta(wh),
          cosThetaH2 = cosThetaH*cosThetaH,
          sinThetaH2 = 1.f - cosThetaH2,
          sinThetaH4 = sinThetaH2*sinThetaH2;
    if (cosThetaO < 0 || cosThetaI < 0 || sinThetaH4 < 1e-6f) return 0.f;

    Float normalization = 1.f / (Pi*(1.f + 4.f*alpha2)),
          exponent      = -cosThetaH2 / (sinThetaH2 * alpha2),
          D             = 1.f + 4.f*std::exp(exponent) / sinThetaH4,
          denominator   = 4.f*(cosThetaO + cosThetaI - cosThetaO*cosThetaI);

    return Csheen * D * normalization / denominator;
}

std::string SheenNeubeltPettineo::ToString() const {
    return StringPrintf("[ SheenNeubeltPettineo Csheen: %s alpha: %f]", Csheen.ToString().c_str(), alpha);
}


/*
    type == "conty_kulla"

    Sheen model presented in

        "Production Friendly Microfacet Sheen BRDF"
        by Conty and Kulla

    in the Physically-based Shading course at SIGGRAPH 2017.
 */
class SheenContyKulla : public BxDF {
  public:
    SheenContyKulla(const Spectrum &Csheen, Float alpha)
        : BxDF(BxDFType(BSDF_REFLECTION | BSDF_GLOSSY)), Csheen(Csheen), alpha(alpha) {}
    Spectrum f(const Vector3f &wo, const Vector3f &wi) const;
    std::string ToString() const;

  private:
    Spectrum Csheen;
    Float alpha;
};

Spectrum SheenContyKulla::f(const Vector3f &wo, const Vector3f &wi) const {
    Vector3f wh = wi + wo;
    if (wh.x == 0 && wh.y == 0 && wh.z == 0) return Spectrum(0.);
    wh = Normalize(wh);

    Float invAlpha = 1.f / alpha;

    Float cosThetaO  = CosTheta(wo),
          cosThetaI  = CosTheta(wi),
          cosThetaH  = CosTheta(wh),
          cosThetaH2 = cosThetaH*cosThetaH,
          sinThetaH2 = 1.f - cosThetaH2;

    if (cosThetaO < 0 || cosThetaI < 0) return 0.f;

    Float D = (2.f + invAlpha) * std::pow(sinThetaH2, 0.5f*invAlpha) * Inv2Pi;

    auto L = [&](Float x) {
        Float r = 1.f - (1.f - alpha)*(1.f - alpha);

        Float a = Lerp(r,  25.3245f,  21.5473f),
              b = Lerp(r,  3.32435f,  3.82987f),
              c = Lerp(r,  0.16801f,  0.19823f),
              d = Lerp(r, -1.27393f, -1.97760f),
              e = Lerp(r, -4.85967f, -4.32054f);

        return a / (1.f + b*std::pow(x, c)) + d*x + e;
    };

    Float lambdaI = cosThetaI < 0.5f ? std::exp(L(cosThetaI))
                                     : std::exp(2.f*L(0.5f) - L(1.f - cosThetaI)),
          lambdaO = cosThetaO < 0.5f ? std::exp(L(cosThetaO))
                                     : std::exp(2.f*L(0.5f) - L(1.f - cosThetaO));

    Float G = 1.f / (1.f + lambdaI + lambdaO);

    return Csheen * D * G / (4.f * cosThetaI * cosThetaO);
}

std::string SheenContyKulla::ToString() const {
    return StringPrintf("[ SheenContyKulla Csheen: %s alpha: %f]", Csheen.ToString().c_str(), alpha);
}


/*
    type == "patry"

    Sheen model presented in

        "Samurai Shading in Ghost of Tsushima"
        by Jasmin Patry

    in the Physically-based Shading course at SIGGRAPH 2020.
 */
class SheenPatry : public BxDF {
  public:
    SheenPatry(const Spectrum &Csheen, Float alpha)
        : BxDF(BxDFType(BSDF_REFLECTION | BSDF_GLOSSY)), Csheen(Csheen), alpha(alpha) {}
    Spectrum f(const Vector3f &wo, const Vector3f &wi) const;
    std::string ToString() const;

  private:
    Spectrum Csheen;
    Float alpha;
};

Spectrum SheenPatry::f(const Vector3f &wo, const Vector3f &wi) const {
    Float cosThetaO = CosTheta(wo),
          cosThetaI = CosTheta(wi);
    if (cosThetaO < 0 || cosThetaI < 0) return 0.f;

    Float sigma   = alpha*alpha,
          density = 1.f;
    sggx::Ellipsoid S = sggx::Ellipsoid::fromFiber(Vector3f(0, 0, 1), sigma);
    Float phaseFunction = sggx::evalPhaseSpecular(wo, wi, S);
    /* The `cosThetaI * cosThetaO` term below differs from the equations
       given in the 2020 talk slides and instead comes from
       "The secret of velvety skin" by Koenderink and Pont 2002, which
       is cited as the original source of this model. */
    Float tmp1      = CosTheta(wo + wi),
          tmp2      = tmp1 / (cosThetaI * cosThetaO),
          numerator = 1.f - std::exp(-density * tmp2);

    return Csheen * phaseFunction * numerator / tmp1;
}

std::string SheenPatry::ToString() const {
    return StringPrintf("[ SheenPatry Csheen: %s alpha: %f]", Csheen.ToString().c_str(), alpha);
}


void SheenBaselinesMaterial::ComputeScatteringFunctions(SurfaceInteraction *si,
                                                        MemoryArena &arena,
                                                        TransportMode mode,
                                                        bool allowMultipleLobes) const {
    // Perform bump mapping with _bumpMap_, if present
    if (bumpMap) Bump(bumpMap, si);

    // Evaluate textures for _SheenBaselinesMaterial_ material and allocate BRDF
    si->bsdf = ARENA_ALLOC(arena, BSDF)(*si);
    Spectrum c = Csheen->Evaluate(*si);
    Float a = Clamp(alpha->Evaluate(*si), 0, 1);
    if (!c.IsBlack()) {
        if (type == "burley") {
            si->bsdf->Add(ARENA_ALLOC(arena, SheenBurley)(c));
        } else if (type == "neubelt_pettineo") {
            si->bsdf->Add(ARENA_ALLOC(arena, SheenNeubeltPettineo)(c, a));
        } else if (type == "conty_kulla") {
            si->bsdf->Add(ARENA_ALLOC(arena, SheenContyKulla)(c, a));
        } else if (type == "patry") {
            si->bsdf->Add(ARENA_ALLOC(arena, SheenPatry)(c, a));
        } else {
            Error("SheenBaselinesMaterial: type \"%s\" unknown", type.c_str());
        }
    }
}

SheenBaselinesMaterial *CreateSheenBaselinesMaterial(const TextureParams &mp) {
    std::string type = mp.FindString("type");
    std::shared_ptr<Texture<Spectrum>> Csheen = mp.GetSpectrumTexture("Csheen", Spectrum(1.f));
    std::shared_ptr<Texture<Float>> alpha     = mp.GetFloatTexture("alpha", 0.5f);
    std::shared_ptr<Texture<Float>> bumpMap   = mp.GetFloatTextureOrNull("bumpmap");
    return new SheenBaselinesMaterial(type, Csheen, alpha, bumpMap);
}

}  // namespace pbrt
