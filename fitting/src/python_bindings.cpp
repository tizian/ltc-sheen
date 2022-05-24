#include <python_bindings.h>

#include <vec3.h>
#include <mat3.h>
#include <pcg32.h>

#include <bsdf.h>
#include <bsdfs/diffuse.h>
#include <bsdfs/ltc.h>
#include <bsdfs/sheen_approx.h>
#include <bsdfs/sheen_volume.h>

// Python bindings for all relevant functionality.

PYBIND11_MODULE(ltcsheen, m) {
    m.doc() = "Sheen LTC fitting utilities";

    py::class_<vec3>(m, "vec3", py::buffer_protocol())
        .def(py::init<>())
        .def(py::init([](const py::buffer &b) {
            py::buffer_info info = b.request();

            if (info.format != py::format_descriptor<double>::format())
                throw std::runtime_error("Incompatible format: expected a double array!");

            if (info.ndim != 1)
                throw std::runtime_error("Incompatible buffer dimension!");

            if (info.shape[0] != 3)
                throw std::runtime_error("Incompatible shape!");

            uint8_t *ptr = (uint8_t *) info.ptr;

            vec3 ret;
            memcpy(&ret.data[0], info.ptr, 3*sizeof(double));
            return ret;
        }))

        .def_buffer([](vec3 &v) -> py::buffer_info {
            return py::buffer_info(
                v.data, sizeof(double),
                py::format_descriptor<double>::format(),
                1, { 3 }, { sizeof(double) }
            );
        })
        .def("__str__", &vec3::to_string)
        .def("__repr__", &vec3::to_string);

    py::implicitly_convertible<py::buffer, vec3>();


    py::class_<mat3>(m, "mat3", py::buffer_protocol())
        .def(py::init<>())
        .def(py::init([](const py::buffer &b) {
            py::buffer_info info = b.request();

            if (info.format != py::format_descriptor<double>::format())
                throw std::runtime_error("Incompatible format: expected a double array!");

            if (info.ndim != 2)
                throw std::runtime_error("Incompatible buffer dimension!");

            if (info.shape[0] != 3 || info.shape[1] != 3)
                throw std::runtime_error("Incompatible shape!");

            uint8_t *ptr = (uint8_t *) info.ptr;

            mat3 ret;
            memcpy(&ret.data[0], info.ptr, 9*sizeof(double));
            return ret;
        }))
        .def_buffer([](mat3 &m) -> py::buffer_info {
            return py::buffer_info(
                m.data, sizeof(double),
                py::format_descriptor<double>::format(),
                2, { 3, 3 }, { 3 * sizeof(double), sizeof(double) }
            );
        })
        .def("__str__", &mat3::to_string)
        .def("__repr__", &mat3::to_string);

    py::implicitly_convertible<py::buffer, mat3>();


    m.def("spherical_direction", spherical_direction);
    m.def("spherical_coordinates", spherical_coordinates);
    m.def("coordinate_system", coordinate_system);

    py::class_<Sampler>(m, "Sampler")
        .def(py::init<>())
        .def("seed", [](Sampler &s, uint64_t seed) {
            s.seed(seed, PCG32_DEFAULT_STREAM + seed);
        }, "seed"_a)
        .def("nextDouble", &Sampler::nextDouble);


    py::class_<Brdf>(m, "Brdf")
        .def("eval", &Brdf::eval,
             "theta_i"_a, "phi_i"_a, "theta_o"_a, "phi_o"_a, "n_evals"_a=1, "sampler_"_a=nullptr)

        .def("eval_vectorized", py::vectorize(&Brdf::eval),
             "theta_i"_a, "phi_i"_a, "theta_o"_a, "phi_o"_a, "n_evals"_a=1, "sampler"_a=nullptr)

        .def("sample", &Brdf::sample,
             "theta_i"_a, "phi_i"_a, "sampler_"_a)

        .def("sample_vectorized", [](const Brdf &brdf,
                                     double theta_i,
                                     double phi_i,
                                     Sampler *sampler,
                                     int n_samples) {
            py::array_t<double> weight(n_samples),
                                theta_o(n_samples),
                                phi_o(n_samples);

            double *weight_ptr = (double *) weight.request().ptr,
                   *theta_o_ptr = (double *) theta_o.request().ptr,
                   *phi_o_ptr = (double *) phi_o.request().ptr;

            for (int i = 0; i < n_samples; ++i) {
                auto [w, t, p] = brdf.sample(theta_i, phi_i, sampler);
                weight_ptr[i] = w;
                theta_o_ptr[i] = t;
                phi_o_ptr[i] = p;
            }

            return std::make_tuple(weight, theta_o, phi_o);
        }, "theta_i"_a, "phi_i"_a, "sampler"_a, "n_samples"_a=1)

        .def("__str__", &Brdf::to_string)
        .def("__repr__", &Brdf::to_string);

    py::class_<DiffuseBrdf, Brdf>(m, "DiffuseBrdf")
        .def(py::init<double>(),
             "reflectance"_a)
        .def_readwrite("reflectance", &DiffuseBrdf::reflectance);

    py::class_<LTCBrdf, Brdf>(m, "LTCBrdf")
        .def(py::init<double, const mat3 &>(),
             "magnitude"_a, "inv_transform"_a)
        .def_readwrite("magnitude", &LTCBrdf::magnitude)
        .def_readwrite("inv_transform", &LTCBrdf::inv_transform);

    py::class_<ApproxSheenBrdf, Brdf>(m, "ApproxSheenBrdf")
        .def(py::init<double>(),
             "alpha"_a)
        .def_readwrite("alpha", &ApproxSheenBrdf::alpha);

    py::class_<VolumeSheenBrdf, Brdf>(m, "VolumeSheenBrdf")
        .def(py::init<int, double, double, double, double>(),
             "maxBounces"_a, "albedo"_a, "density"_a, "sigma"_a, "thetaF"_a=0.0)
        .def_readwrite("maxBounces", &VolumeSheenBrdf::maxBounces)
        .def_readwrite("albedo", &VolumeSheenBrdf::albedo)
        .def_readwrite("density", &VolumeSheenBrdf::density)
        .def_readwrite("sigma", &VolumeSheenBrdf::sigma);
}
