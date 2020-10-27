#include <benchmark/benchmark.h>

#include "cpu.hh"
#include "load.hh"
#include "GPU/gpu.hh"

void BM_CPU_Find_corresponding(benchmark::State& st)
{
    double min_coord_ref[3];
    double max_coord_ref[3];
    double min_coord_scene[3];
    double max_coord_scene[3];
    auto matrix_ref = cpu_load_matrix("data_students/cow_ref.txt", min_coord_ref, max_coord_ref);
    auto matrix_scene = cpu_load_matrix("data_students/cow_tr1.txt", min_coord_scene, max_coord_scene);
    CPU::ICP icp(matrix_ref, matrix_scene, 20);

    for (auto _ : st)
        icp.find_corresponding();

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

void BM_GPU_Find_corresponding(benchmark::State& st)
{
    double min_coord_ref[3];
    double max_coord_ref[3];
    double min_coord_scene[3];
    double max_coord_scene[3];
    auto matrix_ref = load_matrix("data_students/cow_ref.txt", min_coord_ref, max_coord_ref);
    auto matrix_scene = load_matrix("data_students/cow_tr1.txt", min_coord_scene, max_coord_scene);
    GPU::ICP icp(matrix_ref, matrix_scene, 20);

    for (auto _ : st)
        icp.find_corresponding();

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}


BENCHMARK(BM_CPU_Find_corresponding)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK(BM_GPU_Find_corresponding)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK_MAIN();
