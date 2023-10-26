$root_folder = 'c:\IsKnown_Images_IsVisible\Val'

del $out_csv
Get-ChildItem $root_folder |
ForEach-Object {
    #$newname = $_.Parent.FullName + "\" + "99" + $_.Name + "00000"
    $newname = $_.Parent.FullName + "\" + "99" + $_.Name.Substring(70,5)+ "00000"
    echo  $_.FullName
    echo  $newname
    Rename-Item $_.FullName $newname
}
