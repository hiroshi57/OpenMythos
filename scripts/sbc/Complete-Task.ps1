<#
  自動生成ラッパー。編集しないこと。
  second-brain-controller ( $env:SBC_HOME もしくはインストール時のパス ) の Complete-Task.ps1 を、
  このリポジトリ(project=OpenMythos)をコントラクトストアとして呼び出す。
  ロジック本体は second-brain-controller 側にあるため、そちらの修正は再インストール不要で反映される。
#>
$sbcHome = if ($env:SBC_HOME) { $env:SBC_HOME } else { "C:\Users\takiz\repos\second-brain-controller" }
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")

if (-not (Test-Path $sbcHome)) {
    throw "second-brain-controller が見つかりません: $sbcHome 。$env:SBC_HOME を設定してください。"
}

Import-Module (Join-Path $sbcHome "scripts\lib\Contract.psm1") -Force
Initialize-ContractStore -RepoRoot $repoRoot

& (Join-Path $sbcHome "scripts\Complete-Task.ps1") @args
