# Script de teste para o endpoint /tts-with-ref

$text = "Olá, este é um teste do XTTS usando voz de referência do David."
$language = "pt"
$audioFile = "david.wav"
$outputFile = "output_david.wav"

Write-Host "🎤 Testando TTS com voz de referência..." -ForegroundColor Cyan
Write-Host "Texto: $text" -ForegroundColor Yellow
Write-Host "Arquivo de referência: $audioFile" -ForegroundColor Yellow

# Verifica se o arquivo existe
if (-not (Test-Path $audioFile)) {
    Write-Host "❌ Erro: Arquivo $audioFile não encontrado!" -ForegroundColor Red
    exit 1
}

# Monta o form-data
$form = @{
    text = $text
    language = $language
    speaker_wav = Get-Item -Path $audioFile
}

try {
    # Inicia cronômetro
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    
    # Faz a requisição
    Write-Host "📡 Enviando requisição para http://localhost:8000/tts-with-ref..." -ForegroundColor Cyan
    
    Invoke-WebRequest -Method Post `
        -Uri "http://localhost:8000/tts-with-ref" `
        -Form $form `
        -OutFile $outputFile
    
    # Para cronômetro
    $stopwatch.Stop()
    
    Write-Host "✅ Áudio gerado com sucesso: $outputFile" -ForegroundColor Green
    Write-Host "⏱️  Tempo de geração: $($stopwatch.Elapsed.TotalSeconds.ToString('0.00')) segundos" -ForegroundColor Yellow
    Write-Host "🔊 Reproduzindo áudio..." -ForegroundColor Cyan
    
    # Reproduz o áudio (opcional)
    Start-Process $outputFile
    
} catch {
    Write-Host "❌ Erro na requisição:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
