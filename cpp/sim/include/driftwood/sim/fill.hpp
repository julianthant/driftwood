#pragma once

#include <string>
#include <vector>

#include "driftwood/sim/path.hpp"

namespace driftwood::sim {

struct CostModel {
    double commission_per_share = 0.0;
    double half_spread_bps = 1.0;
    double impact_coef = 0.0;
    double participation_limit = 0.1;
};

struct BinFill {
    int bin = 0;
    double raw_price = 0.0;
    double fill_price = 0.0;
    double quantity = 0.0;
    double bin_volume = 0.0;
    double participation = 0.0;
};

struct Fill {
    double filled = 0.0;
    double unfilled = 0.0;
    double cash = 0.0;
    double raw_vwap = 0.0;
    double fill_vwap = 0.0;
    std::vector<BinFill> bins;
};

// `side` is "buy" or "sell". `algo` is "immediate", "twap", or "vwap".
// Quantity is a positive share count. Zero shares is an empty fill.
// Negative shares, a broken bar, or a bad cost is an invalid_argument.
//
// Immediate puts the whole order in the opening bin and prices that bin at
// the open. TWAP targets an equal share count in every bin. VWAP targets
// shares in proportion to the volume profile. Each bin then fills at most
// `participation_limit` of that bin's volume. Whatever does not fit is
// carried into the next bin. Anything still left after the close is unfilled.
//
// The raw price is the path. A buy pays more than that, and a sell receives
// less, by half the spread plus `impact_coef * sigma * sqrt(participation)`.
// Sigma is the caller's trailing volatility, known before this session.
// Commission is charged per filled share in the cash amount. It is not
// inside the fill price. Cash is from the trader's side: a buy is negative.
Fill simulate(
    const Bar& bar,
    double quantity,
    const std::string& side,
    const std::string& algo,
    const CostModel& costs,
    double sigma,
    int n_bins);

}  // namespace driftwood::sim
