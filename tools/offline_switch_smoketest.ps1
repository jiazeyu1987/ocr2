param(
    [string]$HostName = "127.0.0.1",
    [int]$Port = 30415,
    [string]$Code = "31415",
    [int]$PointId1 = 65408,
    [int]$PointId2 = 65409,
    [double]$TimeOutSeconds = 100,
    [bool]$IsSave = $true,
    [int]$StartToStopDelayMs = 300,
    [int]$StopToNextStartDelayMs = 200
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Send-RequestLine {
    param(
        [Parameter(Mandatory = $true)][System.IO.StreamWriter]$Writer,
        [Parameter(Mandatory = $true)][System.IO.StreamReader]$Reader,
        [Parameter(Mandatory = $true)][string]$Type,
        [Parameter(Mandatory = $true)][string]$ArgJson
    )

    $line = "$Type;$Code;$ArgJson"
    $Writer.WriteLine($line)
    $respLine = $Reader.ReadLine()
    if ([string]::IsNullOrWhiteSpace($respLine)) {
        throw "Empty response for request: $line"
    }
    return ($respLine | ConvertFrom-Json)
}

$client = New-Object System.Net.Sockets.TcpClient
try {
    $client.Connect($HostName, $Port)
    $stream = $client.GetStream()
    $writer = New-Object System.IO.StreamWriter($stream, [System.Text.Encoding]::UTF8)
    $writer.AutoFlush = $true
    $reader = New-Object System.IO.StreamReader($stream, [System.Text.Encoding]::UTF8)

    $arg1 = @{ point_id = $PointId1; time_out = $TimeOutSeconds; is_save = $IsSave } | ConvertTo-Json -Compress
    $r1 = Send-RequestLine -Writer $writer -Reader $reader -Type "OFFLINE" -ArgJson $arg1
    "OFFLINE start 1 => $($r1 | ConvertTo-Json -Compress)"

    Start-Sleep -Milliseconds $StartToStopDelayMs

    $r2 = Send-RequestLine -Writer $writer -Reader $reader -Type "OFFLINE" -ArgJson $arg1
    "OFFLINE stop 1  => $($r2 | ConvertTo-Json -Compress)"

    Start-Sleep -Milliseconds $StopToNextStartDelayMs

    $arg2 = @{ point_id = $PointId2; time_out = $TimeOutSeconds; is_save = $IsSave } | ConvertTo-Json -Compress
    $r3 = Send-RequestLine -Writer $writer -Reader $reader -Type "OFFLINE" -ArgJson $arg2
    "OFFLINE start 2 => $($r3 | ConvertTo-Json -Compress)"

    if (-not $r1.success -or -not $r2.success -or -not $r3.success) {
        throw "One or more OFFLINE responses had success=false"
    }
} finally {
    try { $client.Close() } catch { }
}

