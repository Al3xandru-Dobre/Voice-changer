#include "util/WorkerPriority.hpp"
#include "TestSupport.hpp"

#include <string_view>

#if defined(__APPLE__)
#include <pthread.h>
#include <pthread/qos.h>
#include <sys/qos.h>
#include <thread>
#endif

int main() {
    // CI must not depend on RT scheduling privileges or mutate its own QoS.
    const fluxvoice::WorkerPriority disabled(false);
    CHECK(!disabled.elevated());
    CHECK(disabled.result() == 0);
    CHECK(std::string_view(disabled.method()) == "not requested");

#if defined(__APPLE__)
    // BUG-019: QoS needs no privilege, so exercise elevation and restore on a
    // scratch thread (the main thread's QoS is never changed).
    std::thread([] {
        CHECK(pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0) == 0);
        qos_class_t before = QOS_CLASS_UNSPECIFIED;
        int before_rel = 0;
        CHECK(pthread_get_qos_class_np(pthread_self(), &before, &before_rel) == 0);
        CHECK(before == QOS_CLASS_UTILITY);
        {
            const fluxvoice::WorkerPriority prio;
            CHECK(std::string_view(prio.method()) == "QoS user-interactive");
            CHECK(prio.elevated());
            qos_class_t during = QOS_CLASS_UNSPECIFIED;
            int during_rel = 0;
            CHECK(pthread_get_qos_class_np(pthread_self(), &during, &during_rel) == 0);
            CHECK(during == QOS_CLASS_USER_INTERACTIVE);
        }
        qos_class_t after = QOS_CLASS_UNSPECIFIED;
        int after_rel = 0;
        CHECK(pthread_get_qos_class_np(pthread_self(), &after, &after_rel) == 0);
        CHECK(after == before);
        CHECK(after_rel == before_rel);
    }).join();
#endif
    return test::finish("worker_priority");
}
