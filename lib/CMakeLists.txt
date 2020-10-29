include(FetchContent)
FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG bcbaad6d874d451817457ae0603f953cda3c0c06
)
FetchContent_MakeAvailable(Eigen3)
