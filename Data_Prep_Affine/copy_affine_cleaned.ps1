
$src_folder = 'A:\IsKnown_Images\Affine'
$filenames_folder = 'A:\IsKnown_Images\Cleaned'

$dest_folder = 'A:\IsKnown_Images\A_Cleaned\'


Get-ChildItem $filenames_folder -File -Recurse |
Foreach{
    $barcode = $_.DirectoryName.Split('\')[-1]
    #echo $barcode
    $src_filename_full = [system.String]::Join("\",  ($src_folder, $barcode, $_.Name ))
    #echo $src_filename_full
    $dest_folder_full = [system.String]::Join("\",  ($dest_folder, $barcode))

    if (! (Test-Path $dest_folder_full)) {
        md  $dest_folder_full
    }
    xcopy $src_filename_full $dest_folder_full /q /y
}