#include "util/WorkerPriority.hpp"
#include "TestSupport.hpp"

#include <string_view>

int main() {
    // CI must not depend on RT scheduling privileges or mutate its own QoS.
    const fluxvoice::WorkerPriority disabled(false);
    CHECK(!disabled.elevated());
    CHECK(disabled.result() == 0);
    CHECK(std::string_view(disabled.method()) == "not requested");
    return test::finish("worker_priority");
}
