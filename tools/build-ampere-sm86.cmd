@echo off
rem Builds the RTX 30-series (SM86) miner: build\ampere-sm86\canary-miner.exe
rem Needs Visual Studio 2022 (C++ workload), CUDA 12.x, CMake 3.28+ and Ninja on PATH.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cd /d "%~dp0.." || exit /b 1
if not exist build\ampere-sm86\build.ninja (
  cmake -S . -B build\ampere-sm86 -G Ninja -DCANARY_PROFILE=ampere-sm86 -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe" || exit /b 1
)
cmake --build build\ampere-sm86 --target canaryminer
