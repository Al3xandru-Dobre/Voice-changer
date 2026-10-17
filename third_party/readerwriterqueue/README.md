# readerwriterqueue/

moodycamel's lock-free single-producer/single-consumer queue — moves audio
samples between the real-time callback and the inference worker without
mutexes. Upstream: <https://github.com/cameron314/readerwriterqueue>

Fetched automatically by CMake `FetchContent` (tag `v1.0.7`). To vendor
instead, drop `readerwriterqueue.h` + `atomicops.h` here and configure with
`-DFLUXVOICE_FETCH_DEPS=OFF`.
