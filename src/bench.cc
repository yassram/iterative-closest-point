#include <benchmark/benchmark.h>

#include "cpu.hh"
#include "load.hh"
#include "GPU/gpu.hh"

//structs
struct closest_matrix_params_cpy
{
    unsigned int dim;
    unsigned int np;
    unsigned int nm;
    const MatrixXd p;
    const MatrixXd m;
};

struct gpu_closest_matrix_params_cpy
{
    const GPU::Matrix p;
    const GPU::Matrix m;
    GPU::Matrix y;
};

struct err_compute_params_cpy
{
    unsigned int np;
    MatrixXd p;
    double s;
    const MatrixXd r;
    const MatrixXd t;
    const MatrixXd Y;
};

struct gpu_err_compute_params_cpy
{
    const GPU::Matrix Y;
    GPU::Matrix p;
    bool s;
    const GPU::Matrix sr;
    const GPU::Matrix t;
};

struct cpu_centroid_params_cpy
{
    const MatrixXd p;
    const MatrixXd Y;
};

struct gpu_centroid_params_cpy
{
    const GPU::Matrix p;
    const GPU::Matrix Y;
};

struct err_compute_alignment_params_cpy
{
    unsigned int np;
    const MatrixXd p;
    /* const MatrixXd sr; */
    double s;
    const MatrixXd r;
    const MatrixXd t;
    const MatrixXd Y;
};

//Benchmark functions
void BM_CPU_Find_corresponding(benchmark::State &st, Eigen::MatrixXd ref, Eigen::MatrixXd scene, unsigned int ite)
{
    for (auto state : st)
    {
        CPU::ICP icp(ref, scene, ite);
        icp.find_corresponding();
    }

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

void BM_GPU_Find_corresponding(benchmark::State &st, GPU::Matrix ref, GPU::Matrix scene, unsigned int ite)
{

    for (auto state : st)
    {
        GPU::ICP icp(ref, scene, ite);
        icp.find_corresponding();
    }

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

void BM_CPU_closest_matrix(benchmark::State &st, struct closest_matrix_params_cpy params)
{
    for (auto state : st)
    {
        struct CPU::closest_matrix_params params_
            {
                params.dim, params.np, params.nm, params.p, params.m
            };
        CPU::closest_matrix(params_);
    }

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

void BM_CPU_Err_compute(benchmark::State &st, struct err_compute_params_cpy params)
{
    for (auto state : st)
    {
        struct CPU::err_compute_params params_
            {
                params.np, params.p, params.s, params.r, params.t, params.Y
            };
        CPU::err_compute(params_);
    }
    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

void BM_GPU_Err_compute(benchmark::State &st, struct gpu_err_compute_params_cpy params)
{
    for (auto state : st)
    {
        const GPU::Matrix Y_ = params.Y;
        GPU::Matrix p_ = params.p;
        bool ip_ = params.s;
        const GPU::Matrix sr_ = params.sr;
        const GPU::Matrix t_ = params.t;
        compute_err_w(Y_, p_, ip_, sr_, t_);
    }
    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

void BM_CPU_Compute_centroid(benchmark::State &st, struct cpu_centroid_params_cpy params)
{
    for (auto state : st)
    {
        auto mu_p = params.p.rowwise().mean();
        auto mu_y = params.Y.rowwise().mean();

        MatrixXd centroid_p = params.p.colwise() - mu_p;
        MatrixXd centroid_Y = params.Y.colwise() - mu_y;
    }
    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

void BM_GPU_Compute_centroid(benchmark::State &st, struct gpu_centroid_params_cpy params)
{
    for (auto state : st)
    {
        GPU::Matrix mu_p{params.p.rowwise().mean()};
        GPU::Matrix mu_y{params.Y.rowwise().mean()};

        GPU::Matrix centroid_p = substract_col_w(params.p, mu_p);
        GPU::Matrix centroid_Y = substract_col_w(params.Y, mu_y);
    }
    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

void BM_CPU_Err_compute_alignment(benchmark::State &st, struct err_compute_alignment_params_cpy params)
{
    for (auto state : st)
    {
        struct CPU::err_compute_alignment_params params_
            {
                params.np, params.p, params.s, params.r, params.t, params.Y
            };
        CPU::err_compute_alignment(params_);
    }
    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

void BM_GPU_closest_matrix(benchmark::State &st, struct gpu_closest_matrix_params_cpy params)
{
    for (auto state : st)
    {
        struct gpu_closest_matrix_params_cpy params_
            {
                params.p, params.m, params.y
            };
        compute_Y_w(params_.p, params_.m, params_.y);
    }

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

void BM_CPU_find_alignment(benchmark::State &st, std::tuple<Eigen::MatrixXd, CPU::ICP> tpl)
{
    for (auto state : st)
    {
        CPU::ICP icp_{std::get<1>(tpl)};
        Eigen::MatrixXd Y_ = {std::get<0>(tpl)};
        icp_.find_alignment(Y_);
    }

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

void BM_GPU_find_alignment(benchmark::State &st, std::tuple<GPU::Matrix, GPU::ICP> tpl)
{
    for (auto state : st)
    {
        GPU::ICP icp_{std::get<1>(tpl)};
        GPU::Matrix Y_ = {std::get<0>(tpl)};
        icp_.find_alignment(Y_);
    }

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

//Params functions

GPU::Matrix gpu_matrix_loader(const char *s)
{
    auto matrix_ref = load_matrix(s);
    return matrix_ref;
}

Eigen::MatrixXd cpu_matrix_loader(const char *s)
{
    auto matrix_ref = cpu_load_matrix(s);
    return matrix_ref;
}

struct closest_matrix_params_cpy closest_matrix_parameter()
{
    auto matrix_ref = cpu_load_matrix("data_students/cow_ref.txt");
    auto matrix_scene = cpu_load_matrix("data_students/cow_tr1.txt");
    CPU::ICP icp(matrix_ref, matrix_scene, 20);

    struct CPU::closest_matrix_params params_
        {
            icp.get_closest_matrix_params()
        };
    struct closest_matrix_params_cpy params
        {
            params_.dim, params_.np, params_.nm, params_.p, params_.m
        };
    return params;
}

struct err_compute_params_cpy err_compute_parameter()
{
    auto matrix_ref = cpu_load_matrix("data_students/cow_ref.txt");
    auto matrix_scene = cpu_load_matrix("data_students/cow_tr1.txt");

    CPU::ICP icp(matrix_ref, matrix_scene, 20);
    MatrixXd Y = CPU::closest_matrix(icp.get_closest_matrix_params());

    auto params_ = icp.get_err_compute_params(Y);
    struct err_compute_params_cpy params
        {
            params_.np, params_.p, params_.s, params_.r, params_.t, params_.Y
        };

    return params;
}

struct gpu_err_compute_params_cpy gpu_err_compute_parameter(bool s)
{
    auto matrix_ref = load_matrix("data_students/cow_ref.txt");
    auto matrix_scene = load_matrix("data_students/cow_tr1.txt");

    GPU::ICP icp(matrix_ref, matrix_scene, 20);
    GPU::Matrix Y{GPU::Matrix::Zero(icp.getDim(), icp.getNp())};

    auto params_ = icp.get_err_compute_params(Y, s);
    struct gpu_err_compute_params_cpy params
        {
            params_.Y, params_.p, params_.s, params_.sr, params_.t
        };

    return params;
}

struct cpu_centroid_params_cpy cpu_centroid_parameter()
{
    auto matrix_ref = cpu_load_matrix("data_students/cow_ref.txt");
    auto matrix_scene = cpu_load_matrix("data_students/cow_tr1.txt");

    CPU::ICP icp(matrix_ref, matrix_scene, 20);
    MatrixXd Y = CPU::closest_matrix(icp.get_closest_matrix_params());

    struct cpu_centroid_params_cpy params
        {
            icp.new_p, Y
        };
    return params;
}

struct gpu_centroid_params_cpy gpu_centroid_parameter()
{
    auto matrix_ref = load_matrix("data_students/cow_ref.txt");
    auto matrix_scene = load_matrix("data_students/cow_tr1.txt");

    GPU::ICP icp(matrix_ref, matrix_scene, 20);
    GPU::Matrix Y{GPU::Matrix::Zero(icp.getDim(), icp.getNp())};
    auto params_ = icp.get_closest_matrix_params(Y);
    compute_Y_w(params_.m, params_.p, params_.y);

    struct gpu_centroid_params_cpy params
        {
            params_.p, params_.y
        };
    return params;
}

struct err_compute_alignment_params_cpy err_compute_alignment_parameter()
{
    auto matrix_ref = cpu_load_matrix("data_students/cow_ref.txt");
    auto matrix_scene = cpu_load_matrix("data_students/cow_tr1.txt");

    CPU::ICP icp(matrix_ref, matrix_scene, 20);
    MatrixXd Y = CPU::closest_matrix(icp.get_closest_matrix_params());

    auto params_ = icp.get_err_compute_alignment_params(Y);
    struct err_compute_alignment_params_cpy params
        {
            params_.np, params_.p, params_.s, params_.r, params_.t, params_.y
        };

    return params;
}

struct gpu_closest_matrix_params_cpy gpu_closest_matrix_parameter()
{
    auto matrix_ref = load_matrix("data_students/cow_ref.txt");
    auto matrix_scene = load_matrix("data_students/cow_tr1.txt");
    GPU::ICP icp(matrix_ref, matrix_scene, 20);
    GPU::Matrix Y{GPU::Matrix::Zero(icp.getDim(), icp.getNp())};
    struct GPU::gpu_closest_matrix_params params_
        {
            icp.get_closest_matrix_params(Y)
        };
    struct gpu_closest_matrix_params_cpy params
        {
            params_.p, params_.m, params_.y
        };
    return params;
}

std::tuple<Eigen::MatrixXd, CPU::ICP> find_alignment_parameter()
{
    auto matrix_ref = cpu_load_matrix("data_students/cow_ref.txt");
    auto matrix_scene = cpu_load_matrix("data_students/cow_tr1.txt");
    CPU::ICP icp(matrix_ref, matrix_scene, 20);
    return std::tuple<Eigen::MatrixXd, CPU::ICP>(CPU::closest_matrix(icp.get_closest_matrix_params()), icp);
}

std::tuple<GPU::Matrix, GPU::ICP> gpu_find_alignment_parameter()
{
    auto matrix_ref = load_matrix("data_students/cow_ref.txt");
    auto matrix_scene = load_matrix("data_students/cow_tr1.txt");
    GPU::ICP icp(matrix_ref, matrix_scene, 20);
    GPU::Matrix Y{GPU::Matrix::Zero(icp.getDim(), icp.getNp())};
    return std::tuple<GPU::Matrix, GPU::ICP>(Y, icp);
}

//main_calls
BENCHMARK_CAPTURE(BM_CPU_closest_matrix, cpu_closest_matrix, closest_matrix_parameter())
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK_CAPTURE(BM_GPU_closest_matrix, gpu_closest_matrix, gpu_closest_matrix_parameter())
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK_CAPTURE(BM_CPU_find_alignment, cpu_find_alignment, find_alignment_parameter())
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK_CAPTURE(BM_GPU_find_alignment, gpu_find_alignment, gpu_find_alignment_parameter())
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK_CAPTURE(BM_CPU_Compute_centroid, cpu_compute_centroid, cpu_centroid_parameter())
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK_CAPTURE(BM_GPU_Compute_centroid, gpu_compute_centroid, gpu_centroid_parameter())
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK_CAPTURE(BM_CPU_Err_compute, cpu_err_compute, err_compute_parameter())
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK_CAPTURE(BM_GPU_Err_compute, gpu_err_compute, gpu_err_compute_parameter(false))
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK_CAPTURE(BM_CPU_Err_compute_alignment, cpu_err_compute_alignment, err_compute_alignment_parameter())
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK_CAPTURE(BM_GPU_Err_compute, gpu_err_compute_alignment, gpu_err_compute_parameter(true))
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK_CAPTURE(BM_CPU_Find_corresponding, cpu_loop, cpu_matrix_loader("data_students/cow_ref.txt"), cpu_matrix_loader("data_students/cow_tr1.txt"), 20)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK_CAPTURE(BM_GPU_Find_corresponding, gpu_loop, gpu_matrix_loader("data_students/cow_ref.txt"), gpu_matrix_loader("data_students/cow_tr1.txt"), 20)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK_MAIN();
