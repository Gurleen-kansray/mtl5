<#
.SYNOPSIS
    Windows/MSVC port of run_scaling.sh: multi-core GEMM scaling (#108) across
    native-fast / openblas / mkl, one CSV per backend.

.DESCRIPTION
    Threading is a runtime axis of the same per-backend binary:
      native-fast : MTL5_NUM_THREADS=T
      openblas    : OPENBLAS_NUM_THREADS=T
      mkl         : MKL_NUM_THREADS=T
    For T threads we pin to the first T physical cores -- one logical id per core,
    SMT siblings excluded -- so scaling reflects cores, not hyperthreads. Pinning
    is a Windows affinity BITMASK (OR of one bit per physical core), not taskset.

    BLIS is not built (no native MSVC support). Each CSV's `backend` column is
    labelled "<backend>-t<T>" so analyze_scaling.py can recover (backend, T).

.PARAMETER PCores
    First-logical-processor id of each physical core, in the order to fill. The
    default 0,2,4,... matches an SMT machine whose sibling threads are adjacent
    (verify with topology_probe: physical core -> logical mask).

.PARAMETER Threads
    Thread counts to sweep (default 1,2,4,8).

.PARAMETER Sizes
    GEMM sizes (default "1024,2048").

.EXAMPLE
    pwsh benchmarks/run_scaling.ps1 -Threads 1,2,4,8 -Sizes "1024,2048"
#>
[CmdletBinding()]
param(
    # Comma strings, not [int[]]: passing "1,2,4,8" to an [int[]] param via
    # `powershell -File` coerces it to the single integer 1248 (commas read as
    # digit-group separators). Parse the strings into int lists ourselves.
    [string]$PCores = "0,2,4,6,8,10,12,14",
    [string]$Threads = "1,2,4,8",
    [string]$Sizes = "1024,2048",
    [string[]]$Variants = @("native-fast", "openblas", "mkl"),
    [string]$VcpkgToolchain = $(if ($env:VCPKG_ROOT) { Join-Path $env:VCPKG_ROOT "scripts\buildsystems\vcpkg.cmake" } else { "" }),
    [string]$MklRoot = "C:\Users\tomtz\dev\mkl-venv\Library",
    [string]$OpenBlasRoot = "C:\Users\tomtz\dev\openblas-win",
    [string]$OutDir = "benchmarks\data",
    [int]$Jobs = [Environment]::ProcessorCount
)

if ($Variants.Count -eq 1 -and $Variants[0] -match ',') { $Variants = $Variants[0] -split ',' }
[int[]]$PCoreList  = $PCores  -split ',' | ForEach-Object { [int]$_ }
[int[]]$ThreadList = $Threads -split ',' | ForEach-Object { [int]$_ }

# See run_sweeps.ps1 for why $ErrorActionPreference is left at Continue.
$RepoRoot = Split-Path -Parent $PSScriptRoot
$DataDir  = Join-Path $RepoRoot $OutDir
$LogDir   = Join-Path $DataDir "logs"
New-Item -ItemType Directory -Force -Path $DataDir, $LogDir | Out-Null

# Affinity mask covering the first T physical cores (one logical id per core).
function Mask-ForThreads {
    param([int]$T)
    if ($T -gt $PCoreList.Count) {
        throw "T=$T exceeds the $($PCoreList.Count) physical cores in -PCores; add cores or reduce -Threads."
    }
    $m = 0L
    for ($i = 0; $i -lt $T; $i++) { $m = $m -bor (1L -shl $PCoreList[$i]) }
    return $m
}

function Invoke-Native {
    param([string]$Exe, [string[]]$NativeArgs, [string]$LogBase)
    $p = Start-Process -FilePath $Exe -ArgumentList $NativeArgs -PassThru -NoNewWindow -Wait `
                       -RedirectStandardOutput "$LogBase.out.log" -RedirectStandardError "$LogBase.err.log"
    return $p.ExitCode
}

function Build-Variant {
    param([string]$Dir, [string[]]$CMakeArgs)
    $full = Join-Path $RepoRoot $Dir
    if (Test-Path $full) { Remove-Item -Recurse -Force $full }
    $base = @("-B", $full, "-DMTL5_BUILD_BENCHMARKS=ON", "-DMTL5_BUILD_TESTS=OFF",
              "-DMTL5_BUILD_EXAMPLES=OFF", "-DCMAKE_BUILD_TYPE=Release")
    if ((Invoke-Native "cmake" ($base + $CMakeArgs) (Join-Path $LogDir "$Dir.configure")) -ne 0) {
        throw "configure failed for $Dir (see $LogDir\$Dir.configure.err.log)"
    }
    if ((Invoke-Native "cmake" @("--build", $full, "--config", "Release", "--target", "bench_all", "-j", "$Jobs") (Join-Path $LogDir "$Dir.build")) -ne 0) {
        throw "build failed for $Dir (see $LogDir\$Dir.build.err.log)"
    }
    return (Join-Path $full "benchmarks\Release\bench_all.exe")
}

# Run the thread sweep for one backend, merging per-T CSVs into one file.
function Run-Scaling {
    param([string]$Exe, [string]$Backend, [string]$ThreadVar)
    $out = Join-Path $DataDir "gemm_scaling_${Backend}.csv"
    if (Test-Path $out) { Remove-Item $out }
    $first = $true
    foreach ($T in $ThreadList) {
        $mask = Mask-ForThreads $T
        Set-Item "env:$ThreadVar" "$T"
        $env:OMP_NUM_THREADS = "$T"
        $tmp = Join-Path $DataDir "gemm_scaling_${Backend}_t${T}.csv"
        $log = Join-Path $LogDir  "gemm_scaling_${Backend}_t${T}.out"
        Write-Host ("  {0}  T={1}  affinity mask 0x{2:x}" -f $Backend, $T, $mask)

        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $Exe
        $psi.Arguments = "--suite gemm --sizes $Sizes --label ${Backend}-t${T} --csv `"$tmp`""
        $psi.UseShellExecute = $false
        $psi.RedirectStandardOutput = $true
        $p = [System.Diagnostics.Process]::Start($psi)
        try { $p.ProcessorAffinity = [IntPtr]$mask; $p.PriorityClass = 'High' } catch { Write-Warning $_ }
        $p.StandardOutput.ReadToEnd() | Out-File "$log.log" -Encoding utf8
        $p.WaitForExit()
        if ($p.ExitCode -ne 0) { throw "bench failed ($Backend T=$T); see $log.log" }

        if ($first) { Get-Content $tmp | Set-Content $out; $first = $false }
        else { Get-Content $tmp | Select-Object -Skip 1 | Add-Content $out }
        Remove-Item $tmp -Force
    }
    Write-Host "  -> $out"
}

foreach ($v in $Variants) {
    switch ($v) {
        "native-fast" {
            Write-Host "=== native-fast (MTL5_NUM_THREADS) ==="
            $exe = Build-Variant "build-scaling-native-fast" @(
                "-DMTL5_NATIVE_FAST_GEMM=ON", "-DMTL5_WITH_HIGHWAY=ON", "-DMTL5_NATIVE_ARCH=ON")
            Run-Scaling $exe "native-fast" "MTL5_NUM_THREADS"
        }
        "openblas" {
            $obLib = Join-Path $OpenBlasRoot "lib\libopenblas.lib"
            if (-not (Test-Path $obLib)) { Write-Host "=== openblas: SKIPPED (no $obLib) ==="; break }
            Write-Host "=== openblas (OPENBLAS_NUM_THREADS) ==="
            $env:PATH = "$(Join-Path $OpenBlasRoot 'bin');$env:PATH"
            $exe = Build-Variant "build-scaling-openblas" @(
                "-DMTL5_WITH_BLAS=ON", "-DMTL5_WITH_LAPACK=ON",
                "-DBLAS_LIBRARIES=$obLib", "-DLAPACK_LIBRARIES=$obLib")
            Run-Scaling $exe "openblas" "OPENBLAS_NUM_THREADS"
        }
        "mkl" {
            $mklLib = Join-Path $MklRoot "lib\mkl_rt.lib"
            if (-not (Test-Path $mklLib)) { Write-Host "=== mkl: SKIPPED (no $mklLib) ==="; break }
            Write-Host "=== mkl (MKL_NUM_THREADS) ==="
            $env:PATH = "$(Join-Path $MklRoot 'bin');$env:PATH"
            $exe = Build-Variant "build-scaling-mkl" @(
                "-DMTL5_WITH_BLAS=ON", "-DMTL5_WITH_LAPACK=ON",
                "-DBLAS_LIBRARIES=$mklLib", "-DLAPACK_LIBRARIES=$mklLib")
            Run-Scaling $exe "mkl" "MKL_NUM_THREADS"
        }
        default { Write-Warning "unknown variant '$v', skipping" }
    }
}

Write-Host "`nDone. Analyze with:"
Write-Host "  python benchmarks/analyze_scaling.py $DataDir\gemm_scaling_*.csv --plot $DataDir\gemm_scaling.png"
