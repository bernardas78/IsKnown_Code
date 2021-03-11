# Random sample n files and copy to a dir
#   Data prep same for IsEmpty classifier

$cnt_files_to_copy = 1000

$src_folder = 'A:\IsKnown_Images\Affine'
$dest_folder = 'A:\IsKnown_Images\EmptyNot\NotEmpty'

Write-Host 'Counting files in folder '$src_folder
$cnt_total = (Get-ChildItem $src_folder -Recurse | Measure-Object).Count
Write-Host 'Counted '$cnt_total

# random selection of files to copy
$file_indices = 1..$cnt_total | Get-Random -Count  $cnt_files_to_copy


$cntr = 1
Get-ChildItem $src_folder -Recurse -File |
foreach {
    if ($file_indices.Contains($cntr))
    {
        copy $_.FullName $dest_folder
    }
    $cntr += 1
}
