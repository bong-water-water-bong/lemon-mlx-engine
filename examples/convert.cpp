// Convert + quantize a Qwen3.6-A3B (qwen3_5_moe) bf16/fp16 HF checkpoint to a
// combined MLX model that includes the MTP head, with per-component mixed
// precision. Runs on the engine's own (ROCm) MLX build — no Python required.
//
// The official Qwen/Qwen3.6-35B-A3B checkpoint already carries the MTP head
// inline (mtp.* tensors), so converting it directly yields a single one-file
// model with trunk + MTP head — loadable by lemon-mlx-engine's one-file path.
//
// Mixed precision rationale (see --help):
//   - experts / most linears: --bits (4/6/8) — the bulk of the size.
//   - router gate: --router-bits (default 8) — tiny but picks which experts run;
//     4-bit noise can flip borderline expert selection.
//   - lm_head: --lmhead-bits (default 8) — large vocab output projection.
//   - MTP head (mtp.*): --mtp-bits (default = --bits) — match the trunk so the
//     drafts and the (quantized) trunk agree; over-quantizing relative to the
//     trunk can LOWER acceptance.
//
// Usage:
//   convert <in_bf16_dir> <out_dir> --bits 4 [--router-bits 8] [--lmhead-bits 8]
//           [--mtp-bits N] [--group-size 64]

#include <mlx-lm/common/safetensors.h>
#include <mlx-lm/llm/models/qwen35_moe.h>
#include <mlx/mlx.h>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

namespace mx = mlx::core;
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

struct Opts {
    std::string in_dir, out_dir;
    int bits = 4, router_bits = 8, lmhead_bits = 8, mtp_bits = -1, group_size = 64;
};

// Decide the bit width for a given tensor key, or -1 to leave it unquantized.
int bits_for(const std::string& key, const mx::array& w, const Opts& o) {
    // Only quantize 2D/3D matrices whose contraction dim is group-aligned.
    if (w.ndim() < 2) return -1;
    if (w.shape(-1) % o.group_size != 0 || w.shape(-1) < o.group_size) return -1;
    // Norms / conv / biases are excluded by the shape test above (1D) or by the
    // group-alignment test (conv kernel dim is tiny).
    bool is_mtp = key.rfind("mtp.", 0) == 0 || key.find(".mtp.") != std::string::npos;
    if (is_mtp) return o.mtp_bits;
    // Router gate is "...mlp.gate.weight" (NOT gate_proj / gate_up_proj).
    if (key.find(".gate.weight") != std::string::npos &&
        key.find("gate_proj") == std::string::npos &&
        key.find("gate_up") == std::string::npos)
        return o.router_bits;
    if (key == "lm_head.weight" || key.find("lm_head.weight") != std::string::npos)
        return o.lmhead_bits;
    return o.bits;
}

} // namespace

int main(int argc, char** argv) {
    Opts o;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::stoi(argv[++i]); };
        if (a == "--bits") o.bits = next();
        else if (a == "--router-bits") o.router_bits = next();
        else if (a == "--lmhead-bits") o.lmhead_bits = next();
        else if (a == "--mtp-bits") o.mtp_bits = next();
        else if (a == "--group-size") o.group_size = next();
        else if (a == "--help" || a == "-h") {
            std::cerr << "Usage: convert <in_dir> <out_dir> --bits N "
                         "[--router-bits 8] [--lmhead-bits 8] [--mtp-bits N] "
                         "[--group-size 64]\n";
            return 0;
        }
        else pos.push_back(a);
    }
    if (pos.size() != 2) {
        std::cerr << "Need <in_dir> <out_dir>. See --help.\n";
        return 1;
    }
    o.in_dir = pos[0];
    o.out_dir = pos[1];
    if (o.mtp_bits < 0) o.mtp_bits = o.bits;  // match trunk by default

    std::cerr << "[convert] in=" << o.in_dir << " out=" << o.out_dir
              << " bits=" << o.bits << " router=" << o.router_bits
              << " lm_head=" << o.lmhead_bits << " mtp=" << o.mtp_bits
              << " group_size=" << o.group_size << "\n";

    // 1) Parse the source config and build the model (so sanitize() has the
    //    right architecture for the gate_up split + mtp stash).
    std::ifstream cfg_in(fs::path(o.in_dir) / "config.json");
    json cfg_json;
    cfg_in >> cfg_json;
    auto config = cfg_json.get<mlx_lm::Qwen35MoEConfiguration>();
    mlx_lm::Qwen35MoEModel model(config);

    // 2) Load bf16 weights, run sanitize (splits gate_up in bf16, stashes mtp.*).
    std::cerr << "[convert] loading bf16 weights...\n";
    auto weights = mlx_lm::load_safetensors_from_directory(o.in_dir);
    std::cerr << "[convert] loaded " << weights.size() << " tensors; sanitizing...\n";
    weights = model.sanitize(std::move(weights));

    // Re-attach the stashed mtp.* tensors (sanitize moves them out of `weights`)
    // so they get quantized + saved into the same combined checkpoint. The
    // stashed keys already carry the "mtp." prefix.
    for (const auto& [k, v] : model.mtp_weights()) {
        weights.emplace(k, v);
    }
    std::cerr << "[convert] trunk+mtp tensors to write: " << weights.size() << "\n";

    // 3) Quantize per the mixed-precision policy; copy the rest as-is.
    std::unordered_map<std::string, mx::array> out;
    int nq = 0, nkeep = 0;
    for (auto& [key, w] : weights) {
        int b = bits_for(key, w, o);
        if (b <= 0) {
            out.emplace(key, w);
            ++nkeep;
            continue;
        }
        std::string base = key;
        const std::string suf = ".weight";
        if (base.size() > suf.size() &&
            base.compare(base.size() - suf.size(), suf.size(), suf) == 0)
            base = base.substr(0, base.size() - suf.size());
        auto qr = mx::quantize(w, o.group_size, b);  // {wq, scales, biases}
        out.emplace(base + ".weight", qr[0]);
        out.emplace(base + ".scales", qr[1]);
        out.emplace(base + ".biases", qr[2]);
        ++nq;
    }
    std::cerr << "[convert] quantized " << nq << " weights, kept " << nkeep
              << " as-is; evaluating...\n";
    {
        std::vector<mx::array> all;
        for (auto& [k, v] : out) all.push_back(v);
        mx::eval(all);
    }

    // 4) Write a single shard + config.json (with quantization) + copy aux files.
    fs::create_directories(o.out_dir);
    std::cerr << "[convert] saving safetensors...\n";
    mx::save_safetensors((fs::path(o.out_dir) / "model.safetensors").string(), out,
                         {{"format", "mlx"}});

    // config.json: keep source config, add a quantization block (uniform-ish;
    // group_size + the dominant bits — per-tensor scales/biases carry the rest).
    cfg_json["quantization"] = {{"group_size", o.group_size}, {"bits", o.bits}};
    std::ofstream cfg_out(fs::path(o.out_dir) / "config.json");
    cfg_out << cfg_json.dump(2);
    cfg_out.close();

    for (const char* f : {"tokenizer.json", "tokenizer_config.json", "vocab.json",
                          "merges.txt", "special_tokens_map.json",
                          "generation_config.json", "chat_template.jinja"}) {
        auto src = fs::path(o.in_dir) / f;
        if (fs::exists(src)) fs::copy_file(src, fs::path(o.out_dir) / f,
                                           fs::copy_options::overwrite_existing);
    }
    std::cerr << "[convert] done -> " << o.out_dir << "\n";
    return 0;
}
