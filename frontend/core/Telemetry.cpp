// Telemetry — [stats] line formatting.
#include "core/Telemetry.hpp"

#include <cstdio>
#include <ios>
#include <sstream>
#include <string>

namespace fluxvoice::frontend {

std::string TelemetrySnapshot::queued_ms_text() const {
    if (!queued_valid) return "n/a (independent clocks)";
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.0f ms", queued_ms());
    return buf;
}

std::string TelemetrySnapshot::stats_line(double t_seconds) const {
    std::ostringstream o;
    o.setf(std::ios::fixed);
    o.precision(1);
    o << "[stats] t=" << t_seconds << " frames=" << frames << " overruns=" << input_overruns
      << " underruns=" << output_underruns << " drops=" << output_drops << " queued=" << queued_samples
      << " rss_kb=" << rss_bytes / 1024;
    return o.str();
}

} // namespace fluxvoice::frontend
