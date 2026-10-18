#include "driftwood/sim/path.hpp"

#include <cmath>
#include <stdexcept>

namespace driftwood::sim {
namespace {

void require_bins(int n_bins) {
    if (n_bins < 1 || n_bins > 100000) {
        throw std::invalid_argument("n_bins must be between 1 and 100000");
    }
}

}  // namespace

bool is_up_day(const Bar& bar) { return bar.close >= bar.open; }

std::vector<double> session_anchors(const Bar& bar) {
    if (is_up_day(bar)) {
        return {bar.open, bar.low, bar.high, bar.close};
    }
    return {bar.open, bar.high, bar.low, bar.close};
}

double path_price(const Bar& bar, double t) {
    if (!std::isfinite(t)) {
        throw std::invalid_argument("path time must be finite");
    }
    const std::vector<double> anchors = session_anchors(bar);
    if (t <= 0.0) {
        return anchors[0];
    }
    if (t >= 1.0) {
        return anchors[3];
    }
    const double scaled = t * 3.0;
    int segment = static_cast<int>(scaled);
    if (segment > 2) {
        segment = 2;
    }
    const double local = scaled - static_cast<double>(segment);
    return anchors[static_cast<std::size_t>(segment)] * (1.0 - local) +
           anchors[static_cast<std::size_t>(segment) + 1] * local;
}

std::vector<double> bin_prices(const Bar& bar, int n_bins) {
    require_bins(n_bins);
    std::vector<double> prices;
    prices.reserve(static_cast<std::size_t>(n_bins));
    for (int bin = 0; bin < n_bins; ++bin) {
        const double t = (static_cast<double>(bin) + 0.5) / static_cast<double>(n_bins);
        prices.push_back(path_price(bar, t));
    }
    return prices;
}

std::vector<double> volume_profile(int n_bins) {
    require_bins(n_bins);
    std::vector<double> weights(static_cast<std::size_t>(n_bins));
    double sum = 0.0;
    for (int bin = 0; bin < n_bins; ++bin) {
        const double t = (static_cast<double>(bin) + 0.5) / static_cast<double>(n_bins);
        const double weight = 1.0 + std::abs(2.0 * t - 1.0);
        weights[static_cast<std::size_t>(bin)] = weight;
        sum += weight;
    }
    for (double& weight : weights) {
        weight /= sum;
    }
    return weights;
}

}  // namespace driftwood::sim
