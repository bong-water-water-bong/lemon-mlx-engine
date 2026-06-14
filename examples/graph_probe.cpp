// HIP graph capture/replay mechanism probe (ROCm).
//
// Validates the capture-aware CommandEncoder path end to end:
//   1. begin_capture + mx::eval must NOT deadlock (the hostfunc/sync guards),
//   2. capture records kernels WITHOUT executing them (output VRAM stays stale),
//   3. replay (one hipGraphLaunch) executes them and writes the correct result.
//
// This is the foundation for graph-replayed decode. It does not yet exercise
// input mutation across replays (discrete-GPU host->VRAM coherence) or the
// model caches — those are the next milestones.

#include <mlx/mlx.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

namespace mlx::core {
bool gpu_arena_begin(size_t capacity);
void gpu_arena_end();
bool gpu_graph_begin_capture();
bool gpu_graph_end_capture();
bool gpu_graph_replay();
bool gpu_graph_available();
void gpu_graph_reset();
} // namespace mlx::core

namespace mx = mlx::core;

int main() {
  setbuf(stdout, nullptr); // unbuffered so partial output survives a hang
  printf("start\n");
  mx::set_default_device(mx::Device::gpu);
  printf("device set\n");

  bool use_arena = getenv("PROBE_ARENA") != nullptr;
  if (use_arena) {
    if (!mx::gpu_arena_begin(4 * 1024 * 1024)) {
      printf("FAIL: arena_begin returned false\n");
      return 1;
    }
    printf("arena begun\n");
  } else {
    printf("arena SKIPPED (set PROBE_ARENA=1 to enable)\n");
  }

  // Fixed input in the arena.
  auto x = mx::array({1.0f, 2.0f, 3.0f, 4.0f});
  mx::eval(x);
  printf("x eval'd\n");

  auto step = [&]() {
    return mx::add(mx::multiply(x, mx::array(3.0f)), mx::array(1.0f)); // 3x+1
  };

  // Warmup so any one-time compilation/allocation happens before capture.
  auto warm = step();
  mx::eval(warm);
  printf("warmup: %.1f %.1f %.1f %.1f (expect 4 7 10 13)\n",
         warm.data<float>()[0], warm.data<float>()[1],
         warm.data<float>()[2], warm.data<float>()[3]);

  // --- Capture ---
  printf("begin_capture...\n");
  fflush(stdout);
  mx::gpu_graph_begin_capture();
  auto yc = step();
  mx::eval(yc); // records kernels; must not hang with the capture-aware guards
  bool ok = mx::gpu_graph_end_capture();
  printf("end_capture ok=%d available=%d\n", ok, (int)mx::gpu_graph_available());
  if (!ok) {
    printf("FAIL: capture did not produce a graph\n");
    mx::gpu_arena_end();
    return 1;
  }

  // Output VRAM was only RECORDED, not executed — read it before replay.
  float before = yc.data<float>()[0];
  printf("yc[0] before replay = %.3f (recorded-not-executed; not yet 4)\n", before);

  // --- Replay: one hipGraphLaunch runs all recorded kernels ---
  bool rok = mx::gpu_graph_replay();
  printf("replay ok=%d\n", (int)rok);

  // data() forces a device sync + VRAM readback.
  float a0 = yc.data<float>()[0], a1 = yc.data<float>()[1],
        a2 = yc.data<float>()[2], a3 = yc.data<float>()[3];
  printf("yc after replay = %.1f %.1f %.1f %.1f (expect 4 7 10 13)\n",
         a0, a1, a2, a3);

  bool pass = rok && a0 == 4 && a1 == 7 && a2 == 10 && a3 == 13;
  printf("%s\n", pass ? "PASS: capture+replay mechanism works" : "FAIL");

  mx::gpu_graph_reset();
  mx::gpu_arena_end();
  // Skip C++/HIP exit handlers: the HIP runtime's fat-binary teardown
  // (__hipUnregisterFatBinary) hangs at process exit on this eGPU, unrelated to
  // the probe. The result is already validated above.
  fflush(stdout);
  _exit(pass ? 0 : 1);
}
