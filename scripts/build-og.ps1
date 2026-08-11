Add-Type -AssemblyName System.Drawing

$projectRoot = Split-Path -Parent $PSScriptRoot
$outputPath = Join-Path $projectRoot 'public\og.png'
$sourcePath = Join-Path $projectRoot 'public\media\mechanism.png'
$bitmap = [System.Drawing.Bitmap]::new(1200, 630)
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
$graphics.Clear([System.Drawing.Color]::FromArgb(247, 247, 244))

$gridPen = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(24, 31, 41, 51), 1)
for ($x = 0; $x -le 1200; $x += 60) { $graphics.DrawLine($gridPen, $x, 0, $x, 630) }
for ($y = 0; $y -le 630; $y += 60) { $graphics.DrawLine($gridPen, 0, $y, 1200, $y) }

if (Test-Path $sourcePath) {
  $source = [System.Drawing.Image]::FromFile($sourcePath)
  $attrs = [System.Drawing.Imaging.ImageAttributes]::new()
  $matrix = [System.Drawing.Imaging.ColorMatrix]::new()
  $matrix.Matrix33 = 0.42
  $attrs.SetColorMatrix($matrix)
  $graphics.DrawImage($source, [System.Drawing.Rectangle]::new(730, 116, 440, 350), 0, 0, $source.Width, $source.Height, [System.Drawing.GraphicsUnit]::Pixel, $attrs)
  $attrs.Dispose()
  $source.Dispose()
}

$ink = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(31, 41, 51))
$blue = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(60, 84, 136))
$coral = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(230, 75, 53))
$title = [System.Drawing.Font]::new('Arial', 49, [System.Drawing.FontStyle]::Bold)
$small = [System.Drawing.Font]::new('Arial', 15, [System.Drawing.FontStyle]::Bold)
$body = [System.Drawing.Font]::new('Arial', 22, [System.Drawing.FontStyle]::Regular)
$graphics.FillRectangle($coral, 68, 74, 54, 5)
$graphics.DrawString('FALCON / SYNTHESIS', $small, $blue, 68, 101)
$graphics.DrawString("Trajectory Synthesis`nof a Falcon-Inspired`nFlapping Mechanism", $title, $ink, 62, 158)
$graphics.DrawString('SUPPLEMENTARY MATERIALS  |  KHALIFA UNIVERSITY', $small, $ink, 70, 520)
$graphics.DrawString('Flight  /  Reconstruction  /  Mechanism  /  Optimization', $body, $blue, 68, 558)

$bitmap.Save($outputPath, [System.Drawing.Imaging.ImageFormat]::Png)
$title.Dispose(); $small.Dispose(); $body.Dispose(); $ink.Dispose(); $blue.Dispose(); $coral.Dispose(); $gridPen.Dispose(); $graphics.Dispose(); $bitmap.Dispose()
Write-Output $outputPath
