
$dir = "A:\S3\photo\" 
#$dir = "G:\My Drive\VU\Paper'2021"
cd $dir\..
Get-ChildItem $dir -Recurse -File | Group-Object -Property {$_.FullName.Split("\")[-2]} | ForEach { $str = $_.Name+","+$_.Count; echo $str } > fileCnts_S3.csv
#Get-ChildItem $dir -Recurse -File | ForEach-Object { 
#    $barcode = $_.FullName.Split("\")[-2]
#    {
#        $barcode=
#} 
#> list.csv