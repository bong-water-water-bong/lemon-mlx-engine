// Copyright © 2025
#pragma once

#include <mlx/mlx.h>

namespace mlx_lm {

// Persistent device-position scalar for HIP-graph decode. The model's
// device-position RoPE and attention mask read this fixed-address [1] int32
// array; the decode loop updates its contents in place between graph replays so
// one captured graph advances through positions without re-capture.
mlx::core::array& graph_decode_pos();

// In-place device write of the absolute position via a raw kernel, so the fixed
// address the captured graph baked stays valid (slice_update would reassign it).
void set_graph_decode_pos(int offset);

// Advance the device position in place by delta. LOOP-OWNED: call between graph
// replays, never inside the captured step (an in-graph increment races the pos
// readers — RoPE/mask — as a write-after-read hazard → off-by-one).
void advance_graph_decode_pos(int delta);

// When true, the model does NOT set the position itself — the decode loop owns
// it (set once before capture, bumped between replays). False on the plain
// eager path, where the model keeps it in sync each forward.
bool graph_external_pos();
void set_graph_external_pos(bool on);

// Single source of truth for whether HIP-graph decode is active. Read at cache
// creation (to preallocate a static KV buffer) and in the decode loop. ROCm
// only; always false elsewhere so Mac/CUDA use the plain eager path.
bool graph_decode_enabled();

// True only while the single decode step is being CAPTURED (and during replay,
// when the model doesn't actually run). The model uses in-place recurrent-state
// writes ONLY when capturing — during eager warmup it keeps normal reassignment
// (in-place aliasing under the eager async pipeline is unstable). Set by the
// decode loop around begin_capture.
bool graph_capturing();
void set_graph_capturing(bool on);

} // namespace mlx_lm
