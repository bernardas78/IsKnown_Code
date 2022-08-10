$root_folder = 'A:\S3\photo'

$barcodes = Import-Csv -Path 'selected.csv' -Header "barcode" | ForEach { $_.barcode }

ForEach ($barcode in $barcodes)
{
    $pattern = "*"+$barcode+"*"
    Get-ChildItem $root_folder -Recurse -File | Where { $_.FullName -like $pattern} | Select-Object -First 1 |
    ForEach {
        echo $_.Name
    }
}
