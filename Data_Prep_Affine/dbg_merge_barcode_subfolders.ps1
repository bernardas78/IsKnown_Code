# copy all files from \<barcode>\*.jpg to \*.jpg

$src_folder = 'A:\IsKnown_Images\Selected\23'
$dst_folder = 'A:\IsKnown_Images\Selected_Debug\23_folderless'

Get-ChildItem $src_folder -File -Recurse |
Foreach{
    #echo $_.Name
    if ($_.Name -gt '20201210')
    {
        copy $_.FullName $dst_folder
    }
}