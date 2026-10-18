#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <utility>

#include "driftwood/sim/fill.hpp"
#include "driftwood/sim/path.hpp"

namespace py = pybind11;

namespace {

template <typename Result, typename... Args>
auto translated(Result (*func)(Args...)) {
    return [func](Args... args) -> Result {
        try {
            return func(std::forward<Args>(args)...);
        } catch (const std::invalid_argument& exc) {
            throw py::value_error(exc.what());
        }
    };
}

}  // namespace

PYBIND11_MODULE(_sim, module) {
    module.doc() =
        "Deterministic daily-bar execution simulator. Fills are computed here, not in Python.";

    py::class_<driftwood::sim::BinFill>(module, "BinFill")
        .def_readonly("bin", &driftwood::sim::BinFill::bin)
        .def_readonly("raw_price", &driftwood::sim::BinFill::raw_price)
        .def_readonly("fill_price", &driftwood::sim::BinFill::fill_price)
        .def_readonly("quantity", &driftwood::sim::BinFill::quantity)
        .def_readonly("bin_volume", &driftwood::sim::BinFill::bin_volume)
        .def_readonly("participation", &driftwood::sim::BinFill::participation);

    py::class_<driftwood::sim::Fill>(module, "Fill")
        .def_readonly("filled", &driftwood::sim::Fill::filled)
        .def_readonly("unfilled", &driftwood::sim::Fill::unfilled)
        .def_readonly("cash", &driftwood::sim::Fill::cash)
        .def_readonly("raw_vwap", &driftwood::sim::Fill::raw_vwap)
        .def_readonly("fill_vwap", &driftwood::sim::Fill::fill_vwap)
        .def_readonly("bins", &driftwood::sim::Fill::bins);

    module.def(
        "session_anchors",
        [](double open, double high, double low, double close) {
            try {
                return driftwood::sim::session_anchors(
                    driftwood::sim::Bar{open, high, low, close, 0.0});
            } catch (const std::invalid_argument& exc) {
                throw py::value_error(exc.what());
            }
        },
        py::arg("open"), py::arg("high"), py::arg("low"), py::arg("close"));

    module.def(
        "bin_prices",
        [](double open, double high, double low, double close, int n_bins) {
            try {
                return driftwood::sim::bin_prices(
                    driftwood::sim::Bar{open, high, low, close, 0.0}, n_bins);
            } catch (const std::invalid_argument& exc) {
                throw py::value_error(exc.what());
            }
        },
        py::arg("open"), py::arg("high"), py::arg("low"), py::arg("close"), py::arg("n_bins") = 8);

    module.def("volume_profile", translated(&driftwood::sim::volume_profile), py::arg("n_bins") = 8);

    module.def(
        "simulate",
        [](double open, double high, double low, double close, double volume, double quantity,
           const std::string& side, const std::string& algo, double commission_per_share,
           double half_spread_bps, double impact_coef, double participation_limit, double sigma,
           int n_bins) {
            try {
                const driftwood::sim::CostModel costs{
                    commission_per_share, half_spread_bps, impact_coef, participation_limit};
                return driftwood::sim::simulate(
                    driftwood::sim::Bar{open, high, low, close, volume}, quantity, side, algo, costs,
                    sigma, n_bins);
            } catch (const std::invalid_argument& exc) {
                throw py::value_error(exc.what());
            }
        },
        py::arg("open"), py::arg("high"), py::arg("low"), py::arg("close"), py::arg("volume"),
        py::arg("quantity"), py::arg("side"), py::arg("algo"), py::arg("commission_per_share") = 0.0,
        py::arg("half_spread_bps") = 1.0, py::arg("impact_coef") = 0.0,
        py::arg("participation_limit") = 0.1, py::arg("sigma") = 0.0, py::arg("n_bins") = 8);
}
