Set-Location 'C:\Users\AIUSER\Downloads\CASK'
while (Get-Process -Id 46188 -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 15
}
& '.\.venv313\Scripts\python.exe' 'scripts\run_h100_fidelity_overnight.py' --reasoning-datasets --longbench-tasks qasper multi_news hotpotqa musique 2wikimqa --budgets 256 384 --frontier-tag h100_promptheavy_twostage_rerun_20260411 --reference-parallel 1 --replay-runner-parallel 1 --longbench-replay-job-parallel 2 --longbench-count-prompt-tokens true --longbench-slack-budget-trigger true --longbench-allow-prefill-compression false *> 'C:\Users\AIUSER\Downloads\CASK\experiments\frontier\Qwen3-8B\h100_promptheavy_twostage_rerun_20260411\pipeline.log'
