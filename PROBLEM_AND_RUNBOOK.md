# Problem and Runbook (EDABK_SNN_CIM)

## Problems faced so far (with current status)

1. Tier2 cocotb zero-spike failure when using a negative read stimulus word.
	- What happened: The test drove 0xFFFF into `wbs_dat_i[15:0]`, which is signed -1 in the TB core. The LIF accumulator integrated downward, keeping V_mem below the threshold and producing zero spikes in both polarities.
	- Why it matters: The test incorrectly appeared to prove a mapping bug, when the stimulus itself was suppressing spiking.
	- Current: Not active when the stimulus word is a positive magnitude (e.g., 0x0010). If 0xFFFF is used again, the failure will reproduce.

## How to check the problem quickly

### 1) Check Git remote

```powershell
git remote -v
```

Expected: `origin` should be:

- `https://github.com/Nikunj2608/EDABK_SNN_CIM.git` (fetch)
- `https://github.com/Nikunj2608/EDABK_SNN_CIM.git` (push)

### 2) Check what will be committed

```powershell
git status --short
```

Confirm that generated or local-only files are not staged:

- `LEARNING_PATH.md`
- `verilog/tb/sim_build/`
- `verilog/tb/results.xml`
- `verilog/tb/waveform.vcd`
- `verilog/tb/stimuli_class2.txt`

### 3) Check Python/test prerequisites (optional)

```powershell
python --version
pip --version
```

If needed, install dependencies from docs:

```powershell
pip install -r docs/requirements.txt
```

## How to run and verify

### Build (MSBuild task)

```powershell
msbuild /property:GenerateFullPaths=true /t:build /consoleloggerparameters:NoSummary
```

### Common simulation/test entry points

From repo root:

```powershell
cd verilog/tb
```

Then run your cocotb flow (example pattern):

```powershell
make
```

If you use a custom make target/file:

```powershell
make -f Makefile.lif
```

## Debug checklist when run fails

1. Confirm remote and branch are correct.
2. Run `git status --short` to ensure no accidental staged artifacts.
3. Clean generated simulation output and rerun:

```powershell
Remove-Item -Recurse -Force verilog/tb/sim_build -ErrorAction SilentlyContinue
Remove-Item -Force verilog/tb/results.xml, verilog/tb/waveform.vcd -ErrorAction SilentlyContinue
```

4. Re-run build/test command and inspect first error line.
5. If dependency-related, reinstall requirements and rerun.

## Safe push flow

```powershell
git add .
git status
git commit -m "Update RTL/tb changes and add runbook"
git push origin main
```

Before commit, double-check that local notes and generated artifacts are not listed.
