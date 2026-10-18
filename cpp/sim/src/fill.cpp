#include "driftwood/sim/fill.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace driftwood::sim {
namespace {

void require_finite(double value, const char* name) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(std::string(name) + " must be finite");
    }
}

void validate(const Bar& bar, double quantity, const std::string& side, const std::string& algo,
              const CostModel& costs, double sigma, int n_bins) {
    require_finite(bar.open, "open");
    require_finite(bar.high, "high");
    require_finite(bar.low, "low");
    require_finite(bar.close, "close");
    require_finite(bar.volume, "volume");
    require_finite(quantity, "quantity");
    require_finite(costs.commission_per_share, "commission_per_share");
    require_finite(costs.half_spread_bps, "half_spread_bps");
    require_finite(costs.impact_coef, "impact_coef");
    require_finite(costs.participation_limit, "participation_limit");
    require_finite(sigma, "sigma");

    if (bar.open <= 0.0 || bar.high <= 0.0 || bar.low <= 0.0 || bar.close <= 0.0) {
        throw std::invalid_argument("prices must be positive");
    }
    if (bar.low > bar.high || bar.low > bar.open || bar.low > bar.close || bar.high < bar.open ||
        bar.high < bar.close) {
        throw std::invalid_argument("OHLC is inconsistent");
    }
    if (bar.volume < 0.0) {
        throw std::invalid_argument("volume cannot be negative");
    }
    if (quantity < 0.0) {
        throw std::invalid_argument("quantity cannot be negative");
    }
    if (side != "buy" && side != "sell") {
        throw std::invalid_argument("side must be buy or sell");
    }
    if (algo != "immediate" && algo != "twap" && algo != "vwap") {
        throw std::invalid_argument("algo must be immediate, twap, or vwap");
    }
    if (n_bins < 1 || n_bins > 100000) {
        throw std::invalid_argument("n_bins must be between 1 and 100000");
    }
    if (costs.commission_per_share < 0.0) {
        throw std::invalid_argument("commission_per_share cannot be negative");
    }
    if (costs.half_spread_bps < 0.0) {
        throw std::invalid_argument("half_spread_bps cannot be negative");
    }
    if (costs.impact_coef < 0.0) {
        throw std::invalid_argument("impact_coef cannot be negative");
    }
    if (sigma < 0.0) {
        throw std::invalid_argument("sigma cannot be negative");
    }
    if (costs.participation_limit <= 0.0 || costs.participation_limit > 1.0) {
        throw std::invalid_argument("participation_limit must be in (0, 1]");
    }
}

std::vector<double> targets(const std::string& algo, double quantity, int n_bins,
                            const std::vector<double>& profile) {
    std::vector<double> wanted(static_cast<std::size_t>(n_bins), 0.0);
    if (algo == "immediate") {
        wanted[0] = quantity;
        return wanted;
    }
    if (algo == "twap") {
        const double each = quantity / static_cast<double>(n_bins);
        std::fill(wanted.begin(), wanted.end(), each);
        return wanted;
    }
    for (int bin = 0; bin < n_bins; ++bin) {
        wanted[static_cast<std::size_t>(bin)] = quantity * profile[static_cast<std::size_t>(bin)];
    }
    return wanted;
}

double raw_price_for(const Bar& bar, const std::string& algo, int bin, int n_bins) {
    if (algo == "immediate" && bin == 0) {
        return bar.open;
    }
    const double t = (static_cast<double>(bin) + 0.5) / static_cast<double>(n_bins);
    return path_price(bar, t);
}

}  // namespace

Fill simulate(const Bar& bar, double quantity, const std::string& side, const std::string& algo,
              const CostModel& costs, double sigma, int n_bins) {
    validate(bar, quantity, side, algo, costs, sigma, n_bins);
    Fill result;
    if (quantity == 0.0) {
        return result;
    }

    const std::vector<double> profile = volume_profile(n_bins);
    const std::vector<double> wanted = targets(algo, quantity, n_bins, profile);
    const double half_spread = costs.half_spread_bps / 10000.0;
    const double sign = side == "buy" ? 1.0 : -1.0;
    double carry = 0.0;
    double raw_notional = 0.0;
    double fill_notional = 0.0;

    for (int bin = 0; bin < n_bins; ++bin) {
        const double bin_volume = bar.volume * profile[static_cast<std::size_t>(bin)];
        const double desire = wanted[static_cast<std::size_t>(bin)] + carry;
        const double cap = costs.participation_limit * bin_volume;
        const double filled = std::min(desire, cap);
        carry = desire - filled;
        if (filled <= 0.0) {
            continue;
        }
        const double participation = bin_volume > 0.0 ? filled / bin_volume : 0.0;
        const double impact = costs.impact_coef * sigma * std::sqrt(participation);
        const double slip = half_spread + impact;
        const double raw = raw_price_for(bar, algo, bin, n_bins);
        double price = raw * (1.0 + sign * slip);
        if (price < 0.0) {
            price = 0.0;
        }
        result.bins.push_back(BinFill{bin, raw, price, filled, bin_volume, participation});
        result.filled += filled;
        raw_notional += raw * filled;
        fill_notional += price * filled;
        result.cash -= sign * price * filled;
        result.cash -= costs.commission_per_share * filled;
    }

    result.unfilled = quantity - result.filled;
    if (result.filled > 0.0) {
        result.raw_vwap = raw_notional / result.filled;
        result.fill_vwap = fill_notional / result.filled;
    }
    return result;
}

}  // namespace driftwood::sim
