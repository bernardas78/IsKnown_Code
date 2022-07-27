
$root_folder = 'A:\S3\photo\2022-03-11'

$store = 'VILNIUS_RYGOS'
$sco='SCO25'
$timefrom = '20220202183128-991810000000-sco25-Bananai_1kg-34701630-9'
$timeto   = '20220310123848-991810000000-10232987-15-Bananai__1kg'


$full_path = $root_folder + '\' + $store + '\' + $sco

Get-ChildItem $full_path -Recurse -File | Where {$_.name -gt $timefrom} | Where {$_.name -lt $timeto} |
ForEach {
    echo $_.Name
    del $_.FullName
}