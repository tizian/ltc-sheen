#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace py::literals;

#define PYTHON_EXPORT(name) \
    void python_export_##name(py::module &m)
#define PYTHON_DECLARE(name) \
    extern void python_export_##name(py::module &)
#define PYTHON_IMPORT(name) \
    python_export_##name(m)

template <typename Type> pybind11::handle get_type_handle() {
    return pybind11::detail::get_type_handle(typeid(Type), false);
}
