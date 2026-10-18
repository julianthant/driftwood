#include <pybind11/pybind11.h>

namespace py = pybind11;

// The schedules land in the next change. This module exists so the package
// build, and the CI import of driftwood._sim, have something to compile.
PYBIND11_MODULE(_sim, module) {
    module.doc() = "Driftwood execution simulator.";
    module.def("extension_loaded", []() { return true; });
}
