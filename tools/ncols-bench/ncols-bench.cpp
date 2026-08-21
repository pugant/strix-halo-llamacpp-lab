// ncols-bench — micro-bench del kernel mul_mat_vec (path dmmv) del backend Vulkan.
//
// Misura t(ncols) per C = A x B con A[4096x8192] quantizzato Q4_0_ROCMFP4_FAST
// (tipo dei pesi dei GGUF ROCmFP4-STRIX(_LEAN) di produzione) e B[8192xncols] F32,
// ncols in {1,2,4,8,9,12,14,16}. Serve da sub-gate per lo Stadio 2 di T8
// (patch mul_mat_vec_max_cols=16): vedi piano esperimento
// docs/superpowers/plans/2026-08-21-t8-ncols-bench-experiment.md (workspace lab).
//
// Uso: llama-ncols-bench <max_cols_label> [warmup=10] [iters=50] [runs=5]
//   max_cols_label e' solo un'ETICHETTA della build (8 o 16): la costante
//   effettiva si verifica con GGML_VK_PERF_LOGGER=1 (il nome op contiene _VEC
//   iff n <= mul_mat_vec_max_cols).
//
// Output TSV su stdout (marker TREATMENT greppabile):
//   TREATMENT ncols-bench max_cols=<L> ncols=<n> run=<r> t_us=<mean per iter>
//   SUMMARY   ncols-bench max_cols=<L> ncols=<n> mean_us=<m> std_us=<s>
// Timing: tempo cumulato del gruppo di `iters` compute (sincroni) / iters;
// `runs` gruppi -> mean +/- stddev campionaria. Upload e quantizzazione host
// sono fuori dal timing.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int64_t kM = 4096;  // righe di A (= ne[1])
constexpr int64_t kK = 8192;  // colonne di A (= ne[0], dimensione contratta)

// xorshift64 deterministico: run riproducibili senza librerie esterne.
uint64_t g_rng = 0;

uint32_t next_u32() {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (uint32_t)(g_rng >> 32);
}

// ~N(0,1) approssimata (somma di 4 uniformi): i valori esatti non contano,
// conta la riproducibilita' e una scala ragionevole per le scale UE4M3.
float next_float() {
    float acc = 0.0f;
    for (int i = 0; i < 4; ++i) {
        acc += (float)(next_u32() >> 8) * (1.0f / (float)(1 << 24));
    }
    return acc - 2.0f;
}

void fail(const char * msg) {
    fprintf(stderr, "ncols-bench: FATAL: %s\n", msg);
    exit(1);
}

double mean_of(const std::vector<double> & v) {
    double s = 0.0;
    for (double x : v) s += x;
    return v.empty() ? 0.0 : s / (double)v.size();
}

double sample_stddev(const std::vector<double> & v) {
    if (v.size() < 2) return 0.0;
    const double m = mean_of(v);
    double s = 0.0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / (double)(v.size() - 1));
}

struct BenchPoint {
    ggml_backend_t          backend;
    struct ggml_context * ctx;
    ggml_gallocr_t          galloc;
    struct ggml_cgraph *    graph;
    struct ggml_tensor *    a;
    struct ggml_tensor *    b;
    struct ggml_tensor *    d;
};

// Riferimento CPU dello stesso prodotto, per la sanity numerica (dispatch rotto
// su ncols > 8 nella build vk16 restituirebbe garbage o zeri).
bool cpu_reference(const void * qa, const float * hb, int ncols, std::vector<float> & out) {
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) return false;
    ggml_backend_t cpu = ggml_backend_dev_init(cpu_dev, nullptr);
    if (!cpu) return false;

    struct ggml_init_params ip = { 16 * 1024 * 1024, nullptr, /*no_alloc=*/true };
    struct ggml_context * ctx = ggml_init(ip);
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0_ROCMFP4_FAST, kK, kM);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kK, ncols);
    struct ggml_tensor * d = ggml_mul_mat(ctx, a, b);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, d);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
    if (!ggml_gallocr_alloc_graph(ga, gf)) {
        ggml_gallocr_free(ga);
        ggml_free(ctx);
        ggml_backend_free(cpu);
        return false;
    }
    ggml_backend_tensor_set(a, qa, 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, hb, 0, ggml_nbytes(b));
    ggml_backend_graph_compute(cpu, gf);

    out.resize((size_t)kM * ncols);
    ggml_backend_tensor_get(d, out.data(), 0, ggml_nbytes(d));

    ggml_gallocr_free(ga);
    ggml_free(ctx);
    ggml_backend_free(cpu);
    return true;
}

}  // namespace

int main(int argc, char ** argv) {
    const char * max_cols_label = argc > 1 ? argv[1] : "8";
    const int    warmup         = argc > 2 ? std::atoi(argv[2]) : 10;
    const int    iters          = argc > 3 ? std::atoi(argv[3]) : 50;
    const int    runs           = argc > 4 ? std::atoi(argv[4]) : 5;

    const int ncols_list[] = { 1, 2, 4, 8, 9, 12, 14, 16 };

    // Nota: le iGPU (Strix Halo, `uma: 1`) si registrano come IGPU, non GPU
    // (ggml-vulkan.cpp: is_integrated_gpu -> GGML_BACKEND_DEVICE_TYPE_IGPU).
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    if (!dev) fail("nessun device GPU/IGPU nel registry (backend Vulkan mancante?)");
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) fail("init backend GPU fallito");

    printf("# ncols-bench max_cols=%s warmup=%d iters=%d runs=%d\n", max_cols_label, warmup, iters, runs);
    printf("# backend=%s device=%s (%s)\n", ggml_backend_name(backend), ggml_backend_dev_name(dev),
           ggml_backend_dev_description(dev));
    printf("# A=[%lld x %lld] %s  B=%lld x ncols F32  (K=%lld)\n", (long long)kM, (long long)kK,
           ggml_type_name(GGML_TYPE_Q4_0_ROCMFP4_FAST), (long long)kK, (long long)kK);
    printf("# device=A f32 random deterministica (xorshift64), upload fuori timing\n");
    fflush(stdout);

    // --- dati host (una volta sola, fuori dal timing) ---
    g_rng = 0x9E3779B97F4A7C15ull;
    std::vector<float> ha((size_t)kM * kK);
    for (size_t i = 0; i < ha.size(); ++i) ha[i] = next_float();

    const size_t a_nbytes = (size_t)(kM * kK / ggml_blck_size(GGML_TYPE_Q4_0_ROCMFP4_FAST) *
                                     ggml_type_size(GGML_TYPE_Q4_0_ROCMFP4_FAST));
    std::vector<uint8_t> qa(a_nbytes);
    const size_t qn = ggml_quantize_chunk(GGML_TYPE_Q4_0_ROCMFP4_FAST, ha.data(), qa.data(), 0, kM, kK, nullptr);
    if (qn == 0 || qn != a_nbytes) fail("ggml_quantize_chunk ROCmFP4_FAST fallita");
    ha.clear();
    ha.shrink_to_fit();

    g_rng = 0x85EBCA6B85EBCA6Bull;
    std::vector<float> hb((size_t)kK * 16);
    for (size_t i = 0; i < hb.size(); ++i) hb[i] = next_float();

    std::vector<float> gpu_out;
    gpu_out.reserve((size_t)kM * 16);

    for (int ncols : ncols_list) {
        // --- grafo per questo punto ---
        struct ggml_init_params ip = { 16 * 1024 * 1024, nullptr, /*no_alloc=*/true };
        BenchPoint bp = {};
        bp.backend = backend;
        bp.ctx     = ggml_init(ip);
        if (!bp.ctx) fail("ggml_init fallita");
        bp.a = ggml_new_tensor_2d(bp.ctx, GGML_TYPE_Q4_0_ROCMFP4_FAST, kK, kM);
        bp.b = ggml_new_tensor_2d(bp.ctx, GGML_TYPE_F32, kK, ncols);
        bp.d = ggml_mul_mat(bp.ctx, bp.a, bp.b);
        bp.graph = ggml_new_graph(bp.ctx);
        ggml_build_forward_expand(bp.graph, bp.d);
        bp.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(bp.galloc, bp.graph)) fail("alloc grafo fallita");

        ggml_backend_tensor_set(bp.a, qa.data(), 0, ggml_nbytes(bp.a));
        ggml_backend_tensor_set(bp.b, hb.data(), 0, ggml_nbytes(bp.b));

        // --- warm-up ---
        for (int w = 0; w < warmup; ++w) {
            ggml_backend_graph_compute(backend, bp.graph);
        }

        // --- runs misurati: tempo cumulato del gruppo / iters ---
        std::vector<double> per_run_us;
        for (int r = 0; r < runs; ++r) {
            const int64_t t0 = ggml_time_us();
            for (int i = 0; i < iters; ++i) {
                ggml_backend_graph_compute(backend, bp.graph);
            }
            const int64_t t1 = ggml_time_us();
            const double us  = (double)(t1 - t0) / (double)iters;
            per_run_us.push_back(us);
            printf("TREATMENT ncols-bench max_cols=%s ncols=%d run=%d t_us=%.1f\n", max_cols_label, ncols, r, us);
            fflush(stdout);
        }
        printf("SUMMARY ncols-bench max_cols=%s ncols=%d mean_us=%.1f std_us=%.1f\n", max_cols_label, ncols,
               mean_of(per_run_us), sample_stddev(per_run_us));
        fflush(stdout);

        // --- sanity numerica (ncols 8 e 14): GPU vs CPU ---
        if (ncols == 8 || ncols == 14) {
            gpu_out.resize((size_t)kM * ncols);
            ggml_backend_tensor_get(bp.d, gpu_out.data(), 0, ggml_nbytes(bp.d));

            std::vector<float> ref;
            if (!cpu_reference(qa.data(), hb.data(), ncols, ref)) {
                printf("CHECK ncols=%d SKIP (riferimento CPU non disponibile)\n", ncols);
            } else {
                double sum_abs = 0.0;
                for (float x : ref) sum_abs += std::fabs((double)x);
                const double scale = sum_abs / (double)ref.size() + 1e-12;
                double max_rel = 0.0;
                for (size_t i = 0; i < ref.size(); ++i) {
                    const double rel = std::fabs((double)gpu_out[i] - (double)ref[i]) / scale;
                    if (rel > max_rel) max_rel = rel;
                }
                const bool ok = max_rel < 1e-1;  // MMVQ quantizza B a Q8_1: tolleranza larga
                printf("CHECK ncols=%d max_rel_err=%.4f (scala=%.3f) %s\n", ncols, max_rel, scale, ok ? "OK" : "FAIL");
                fflush(stdout);
                if (!ok) fail("sanity numerica fallita: output GPU diverge dal riferimento CPU");
            }
        }

        ggml_gallocr_free(bp.galloc);
        ggml_free(bp.ctx);
    }

    ggml_backend_free(backend);
    printf("# done\n");
    return 0;
}
