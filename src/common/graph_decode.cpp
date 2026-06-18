// Copyright © 2025
#include "mlx-lm/common/graph_decode.h"
#include <cstdlib>

namespace mx = mlx::core;

// In-place device-scalar kernels (ROCm backend). These mutate the pos buffer's
// CONTENTS without reallocating, so the fixed address a captured HIP graph baked
// in stays valid. slice_update would reassign the buffer and break replay.
namespace mlx::core {
void gpu_kv_pos_set(array& pos, int v);
void gpu_kv_pos_increment(array& pos, int delta);
}

namespace mlx_lm {

static bool g_external = false;
static bool g_capturing = false;

// Constructed lazily on first use, NOT at static-init time. Building it at static
// init (a global mx::array) forces a HIP stream to be created on the default GPU
// before main() runs — i.e. before --device selection — which both strands it on
// device 0 and, on a discrete GPU over TB5, intermittently hangs during process
// startup. A function-local static defers construction to runtime on the selected
// device.
mx::array& graph_decode_pos() {
    static mx::array* g_pos = nullptr;
    if (g_pos == nullptr) {
        g_pos = new mx::array(mx::zeros({1}, mx::int32));
        mx::eval(*g_pos);
    }
    return *g_pos;
}

void set_graph_decode_pos(int offset) {
    // Mutate the pos buffer in place via a raw kernel (NOT slice_update, which
    // reassigns the buffer and would break the captured graph's baked address).
#if defined(MLX_BUILD_ROCM)
    auto& p = graph_decode_pos();
    mx::gpu_kv_pos_set(p, offset);
    mx::synchronize(mx::default_stream(mx::default_device()));
#else
    auto& p = graph_decode_pos();
    p = mx::slice_update(p, mx::broadcast_to(mx::array(offset, mx::int32), p.shape()),
                         mx::Shape(p.ndim(), 0), p.shape());
    mx::eval(p);
#endif
}

// Advance the device position in place by delta (loop-owned, BETWEEN graph
// replays — never inside the captured step, where it would race the pos readers).
void advance_graph_decode_pos(int delta) {
#if defined(MLX_BUILD_ROCM)
    // No synchronize: the increment kernel is launched on the generation stream
    // and the graph replay runs on the same stream right after, so it is ordered
    // after the increment without a host sync (which would kill the pipeline).
    auto& p = graph_decode_pos();
    mx::gpu_kv_pos_increment(p, delta);
#else
    set_graph_decode_pos(0);  // no-op-ish fallback (non-ROCm has no graph path)
#endif
}

bool graph_external_pos() { return g_external; }
void set_graph_external_pos(bool on) { g_external = on; }

bool graph_capturing() { return g_capturing; }
void set_graph_capturing(bool on) { g_capturing = on; }

bool graph_decode_enabled() {
#if defined(MLX_BUILD_ROCM)
    // Opt-in during bring-up; Phase E flips this default to on (MLX_NO_GRAPH off).
    static const bool on = std::getenv("MLX_DECODE_GRAPH") != nullptr;
    return on;
#else
    return false;
#endif
}

} // namespace mlx_lm
