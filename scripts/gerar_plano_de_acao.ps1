#!/usr/bin/env pwsh
<#
.SYNOPSIS
  Regenera docs/plano-de-acao.md a partir das issues/milestones do GitHub.
.DESCRIPTION
  Nao requer autenticacao para repositorio publico. Uso:
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts/gerar_plano_de_acao.ps1
  Board (Projects v2): https://github.com/users/GuilhermeBn198/projects/7
  O script e ASCII-only (compativel com Windows PowerShell 5.1); os acentos da saida
  sao montados por code point para nao depender da codificacao do arquivo.
#>
param(
  [string]$Owner = "GuilhermeBn198",
  [string]$Repo = "ptt6-yolox-edge-bench",
  [string]$Board = "https://github.com/users/GuilhermeBn198/projects/7",
  [string]$OutFile = "$PSScriptRoot/../docs/plano-de-acao.md"
)

$ErrorActionPreference = "Stop"
$headers = @{ "User-Agent" = "ptt6-plano-de-acao"; "Accept" = "application/vnd.github+json" }
$api = "https://api.github.com/repos/$Owner/$Repo"

# acentos via code point (ASCII-safe no arquivo)
$dash = [char]0x2014
$h1     = "Plano de A" + [char]0xE7 + [char]0xE3 + "o"          # Acao (com cedilha/til)
$repoLbl = "Reposit" + [char]0xF3 + "rio"                        # Repositorio
$depH   = "Depend" + [char]0xEA + "ncias"                        # Dependencias
$seqH   = "sequ" + [char]0xEA + "ncia"                           # sequencia
$titLbl = "T" + [char]0xED + "tulo"                              # Titulo

function Get-Paged($url) {
  $all = @()
  $page = 1
  while ($true) {
    $sep = if ($url.Contains("?")) { "&" } else { "?" }
    $items = Invoke-RestMethod -Uri "$url${sep}per_page=100&page=$page" -Headers $headers
    if (-not $items) { break }
    $all += $items
    if ($items.Count -lt 100) { break }
    $page++
  }
  return $all
}

function Get-BlockedBy($num) {
  try {
    $b = Invoke-RestMethod -Uri "$api/issues/$num/dependencies/blocked_by" -Headers $headers |
      ForEach-Object { $_.number } | Sort-Object
    return ($b -join ', ')
  } catch {
    return ""
  }
}

$issues = Get-Paged "$api/issues?state=all" | Where-Object { -not $_.pull_request }
$milestones = Invoke-RestMethod -Uri "$api/milestones?state=all&per_page=100" -Headers $headers
$issues = $issues | Sort-Object number

$msRows = foreach ($m in $milestones) {
  "| $($m.title) | $($m.open_issues) | $($m.closed_issues) | $($m.open_issues + $m.closed_issues) |"
}

$issueRows = foreach ($i in $issues) {
  $state = $i.state.ToUpper()
  $closed = if ($i.closed_at) { $i.closed_at } else { $dash }
  $ms = if ($i.milestone) { $i.milestone.title } else { $dash }
  $labels = (($i.labels | ForEach-Object { $_.name }) -join ', ')
  if (-not $labels) { $labels = $dash }
  $blocked = Get-BlockedBy $i.number
  if (-not $blocked) { $blocked = $dash }
  "| #$($i.number) | $($i.title) | $state | $ms | $labels | $($i.created_at) | $closed | $blocked |"
}

$depLines = foreach ($i in $issues) {
  $b = Get-BlockedBy $i.number
  if ($b) { "#$($i.number) <- $b" }
}

$stamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$nl = "`n"
$content = @"
# $h1 $($dash) PTT6 (registro das issues)

> Gerado a partir do GitHub. $($repoLbl): https://github.com/$Owner/$Repo
> Board (Projects v2): $Board
> Gerado em: $stamp

## Milestones

| Milestone | Abertas | Fechadas | Total |
|---|---:|---:|---:|
$($msRows -join $nl)

## Issues em $seqH

| # | $titLbl | Estado | Milestone | Labels | Criada | Fechada | Bloqueada por |
|---:|---|---|---|---|---|---|---|
$($issueRows -join $nl)

## $depH (blocked by)

``````
$($depLines -join $nl)
``````
"@

[System.IO.File]::WriteAllText((Resolve-Path $OutFile), $content, (New-Object System.Text.UTF8Encoding $false))
Write-Host "Atualizado: $OutFile"
