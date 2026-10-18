# nlohmann_json/

JSON for Modern C++ (MIT). Upstream: <https://github.com/nlohmann/json>

Parses the model contract (`src/pipeline/ModelContract.hpp`, Milestone 2 /
Stage 02) and, later, `voices.json` and voice presets (Stages 08-09).

Fetched automatically by CMake `FetchContent` from the v3.12.0 release asset
`json.tar.xz`, checked against
`SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa`
(the hash in the release notes). Only the headers are used, as the SYSTEM
INTERFACE target `nlohmann_json_headers`. To vendor a copy instead, extract the
tarball's `include/` into this directory (so `include/nlohmann/json.hpp`
exists) and configure with `-DFLUXVOICE_FETCH_DEPS=OFF`.
