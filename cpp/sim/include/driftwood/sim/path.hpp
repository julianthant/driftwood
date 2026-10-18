#pragma once

#include <vector>

namespace driftwood::sim {

struct Bar {
    double open = 0.0;
    double high = 0.0;
    double low = 0.0;
    double close = 0.0;
    double volume = 0.0;
};

// A daily bar has no tape. The path below is a deterministic stand-in that
// stays inside the high and the low. An up day (close >= open) prints
// open, then the low, then the high, then the close. A down day prints
// open, then the high, then the low, then the close. Those four anchors
// sit at times 0, 1/3, 2/3, and 1, and the price between them is linear.
// Bin i of n is the price at the midpoint of that bin, time (i + 0.5) / n.

bool is_up_day(const Bar& bar);

// Four anchor prices in time order.
std::vector<double> session_anchors(const Bar& bar);

double path_price(const Bar& bar, double t);

std::vector<double> bin_prices(const Bar& bar, int n_bins);

// Fractions of the day's volume, one per bin. The shape is a U: the open
// and the close are heavier than the middle. The fractions sum to 1.
std::vector<double> volume_profile(int n_bins);

}  // namespace driftwood::sim
