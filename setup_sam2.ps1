param(
    [string]$VenvPath = ".venv-sam2",
    [ValidateSet("tiny", "small", "base_plus", "large")]
    [string]$Model = "large",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu121"
)

$ErrorActionPreference = "Stop"

function Download-FileIfMissing {
    param(
        [string]$Url,
        [string]$OutputPath
    )

    if (Test-Path -LiteralPath $OutputPath) {
        "Checkpoint already exists: $OutputPath"
        return
    }

    $parent = Split-Path -Parent $OutputPath
    if (-not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Path $parent | Out-Null
    }

    "Downloading $Url"
    Invoke-WebRequest -Uri $Url -OutFile $OutputPath
}

if (-not (Test-Path -LiteralPath ".")) {
    throw "Run this script from the repository root."
}

python -m venv $VenvPath
$pythonExe = Join-Path $VenvPath "Scripts\python.exe"

& $pythonExe -m pip install --upgrade pip setuptools wheel
& $pythonExe -m pip install torch torchvision --index-url $TorchIndexUrl
& $pythonExe -m pip install opencv-python numpy pyyaml
& $pythonExe -m pip install "git+https://github.com/facebookresearch/sam2.git"

$checkpointMap = @{
    tiny = @{
        File = "sam2.1_hiera_tiny.pt"
        Config = "configs/sam2.1/sam2.1_hiera_t.yaml"
        Url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
    }
    small = @{
        File = "sam2.1_hiera_small.pt"
        Config = "configs/sam2.1/sam2.1_hiera_s.yaml"
        Url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
    }
    base_plus = @{
        File = "sam2.1_hiera_base_plus.pt"
        Config = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        Url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
    }
    large = @{
        File = "sam2.1_hiera_large.pt"
        Config = "configs/sam2.1/sam2.1_hiera_l.yaml"
        Url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    }
}

$checkpoint = $checkpointMap[$Model]
Download-FileIfMissing -Url $checkpoint.Url -OutputPath (Join-Path "checkpoints" $checkpoint.File)

""
"SAM2 setup complete."
"Activate with: .\$VenvPath\Scripts\Activate.ps1"
"Run example: python sam2_two_rats.py --video test_100.avi --init-frame 20 --checkpoint checkpoints\$($checkpoint.File)"
"Config inferred by default: $($checkpoint.Config)"
