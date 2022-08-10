$root_folder = 'c:\IsKnown_Images_IsVisible\Test'
$out_csv = 'd:\IsKnown_code\Data_Prep_s3\selected.csv'

del $out_csv
Get-ChildItem $root_folder |
ForEach-Object {
    #$pattern = "*"+$barcode+"*"
    #Get-ChildItem $root_folder -Recurse -File | Where { $_.FullName -like $pattern} | Select-Object -First 1 |
    #ForEach {
    $out_line = "99" + $_.Name + "00000"
    echo  $out_line >> $out_csv
    #}
}
